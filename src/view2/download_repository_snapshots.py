#!/usr/bin/env python3
"""
下载 JSONL 数据集中每个 entry 的 GitHub 仓库 snapshot 和对应的 commit message。
"""

import json
import argparse
import os
import subprocess
import logging
import shutil
import re
import urllib.request
import urllib.error
from pathlib import Path
from typing import Dict, Optional, Tuple
import time
import tempfile

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
logger = logging.getLogger(__name__)


def configure_logging(log_dir: Optional[Path] = None) -> None:
    """Configure logging only when the command is executed."""
    handlers = [logging.StreamHandler()]
    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_dir / "download_repository_snapshots.log", encoding="utf-8"))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=handlers,
    )


def get_github_token() -> Optional[str]:
    """Read optional authentication from the caller's environment."""
    return os.environ.get("GITHUB_TOKEN", "").strip() or None


def get_proxy_settings() -> Tuple[Optional[str], Optional[str]]:
    """Use caller-provided HTTP(S) proxy variables, or connect directly."""
    http_proxy = os.environ.get("http_proxy") or os.environ.get("HTTP_PROXY")
    https_proxy = os.environ.get("https_proxy") or os.environ.get("HTTPS_PROXY")
    return http_proxy, https_proxy


def build_url_opener():
    http_proxy, https_proxy = get_proxy_settings()
    proxies = {}
    if http_proxy:
        proxies["http"] = http_proxy
    if https_proxy:
        proxies["https"] = https_proxy
    return urllib.request.build_opener(urllib.request.ProxyHandler(proxies))


def verify_github_token(token: Optional[str]) -> bool:
    """Check authentication only when explicitly requested by the caller."""
    if not token:
        logger.error("Set GITHUB_TOKEN before using --verify-token.")
        return False
    request = urllib.request.Request("https://api.github.com/user")
    request.add_header("Authorization", f"Bearer {token}")
    request.add_header("Accept", "application/vnd.github+json")
    request.add_header("User-Agent", "CAMGen-replication-package")
    try:
        with build_url_opener().open(request, timeout=15) as response:
            if response.status != 200:
                logger.error("GitHub authentication failed (HTTP %s).", response.status)
                return False
            logger.info("GitHub authentication succeeded.")
            return True
    except urllib.error.HTTPError as error:
        logger.error("GitHub authentication failed (HTTP %s).", error.code)
    except urllib.error.URLError as error:
        logger.error("GitHub authentication request failed: %s", error.reason)
    return False


def download_repository_zip_via_api(owner: str, repo: str, commit_sha: str, zip_file: Path, token: Optional[str] = None) -> bool:
    """
    使用GitHub API下载指定commit的zip文件（最快最直接的方法）
    
    Args:
        owner: 仓库所有者
        repo: 仓库名
        commit_sha: commit SHA
        zip_file: 保存zip文件的路径
        token: GitHub Personal Access Token（可选，但推荐使用以提高速率限制）
        
    Returns:
        bool: 是否下载成功
    """
    # 确保父目录存在
    zip_file.parent.mkdir(parents=True, exist_ok=True)
    
    # 如果文件已存在，先删除
    if zip_file.exists():
        zip_file.unlink()
    
    # 构建GitHub API URL
    api_url = f"https://api.github.com/repos/{owner}/{repo}/zipball/{commit_sha}"
    
    logger.info(f"正在通过GitHub API下载: {api_url} (commit: {commit_sha[:8]}...)")
    
    try:
        req = urllib.request.Request(api_url)
        req.add_header("User-Agent", "CAMGen-replication-package")
        if token:
            req.add_header("Authorization", f"Bearer {token}")
        opener = build_url_opener()

        # 下载文件
        with opener.open(req, timeout=300) as response:
            # 检查响应状态
            if response.status != 200:
                logger.warning(f"API返回状态码: {response.status}")
                return False
            
            total_size = int(response.headers.get('Content-Length', 0))
            downloaded = 0
            chunk_size = 8192
            
            with open(zip_file, 'wb') as f:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    
                    # 每下载10MB输出一次进度
                    if downloaded % (10 * 1024 * 1024) < chunk_size:
                        if total_size > 0:
                            percent = (downloaded / total_size) * 100
                            logger.debug(f"下载进度: {percent:.1f}% ({downloaded//1024//1024}MB/{total_size//1024//1024}MB)")
        
        # 检查文件是否下载成功
        if zip_file.exists() and zip_file.stat().st_size > 0:
            file_size_mb = zip_file.stat().st_size // 1024 // 1024
            logger.info(f"✓ API下载成功: {zip_file.name} ({file_size_mb}MB)")
            return True
        else:
            logger.error(f"API下载失败或文件为空")
            return False
            
    except urllib.error.HTTPError as e:
        try:
            body = e.read().decode('utf-8', errors='replace')[:500] if e.fp else ""
        except Exception:
            body = ""
        # 速率限制是 403，认证失败是 401，两者不同
        if e.code == 401:
            logger.error("GitHub authentication failed (HTTP 401); check GITHUB_TOKEN.")
        elif e.code == 403:
            # 检查是否是速率限制：响应里常有 "rate limit" 或 "API rate limit"
            is_rate_limit = body and ("rate limit" in body.lower() or "rate_limit" in body.lower())
            remaining = e.headers.get("X-RateLimit-Remaining", "")
            limit = e.headers.get("X-RateLimit-Limit", "")
            reset = e.headers.get("X-RateLimit-Reset", "")
            if is_rate_limit or (remaining == "0" and limit):
                reset_msg = f"，重置时间(UTC): {time.strftime('%Y-%m-%d %H:%M', time.gmtime(int(reset)))}" if reset.isdigit() else ""
                logger.error(f"API 速率限制(403): 本小时额度已用尽(剩余 {remaining}/{limit}){reset_msg}。等重置后继续或改用 git clone。")
            else:
                logger.error(f"访问被拒绝(403): 可能是权限不足或其它限制。响应: {body[:200] if body else '无'}")
        elif e.code == 404:
            logger.error(f"仓库或commit不存在(404)")
        else:
            logger.error(f"HTTP错误 {e.code}: {e.reason}")
        return False
    except urllib.error.URLError as e:
        logger.error(f"URL错误: {e.reason}")
        return False
    except Exception as e:
        logger.error(f"API下载出错: {str(e)}")
        return False


def download_repository_zip(owner: str, repo: str, commit_sha: str, zip_file: Path) -> bool:
    """
    下载指定commit的仓库zip文件（优先使用API，失败则使用git clone）
    
    Args:
        owner: 仓库所有者
        repo: 仓库名
        commit_sha: commit SHA
        zip_file: 保存zip文件的路径
        
    Returns:
        bool: 是否下载成功
    """
    zip_file = zip_file.resolve()
    zip_file.parent.mkdir(parents=True, exist_ok=True)
    if zip_file.exists():
        zip_file.unlink()

    github_token = get_github_token()
    if download_repository_zip_via_api(owner, repo, commit_sha, zip_file, github_token):
        return True
    logger.info("API download failed; trying Git clone and archive.")

    # 方法2: 如果API失败或没有token，使用git clone + git archive
    # 构建GitHub仓库URL
    git_url = f"https://github.com/{owner}/{repo}.git"
    
    # Use a unique scratch directory owned by this download operation.
    scratch = tempfile.TemporaryDirectory(prefix="camgen-snapshot-")
    scratch_root = Path(scratch.name).resolve()
    temp_dir = scratch_root / "repository"

    def clear_checkout():
        resolved = temp_dir.resolve()
        if resolved != scratch_root / "repository" or not resolved.is_relative_to(scratch_root):
            raise ValueError("Checkout cleanup target is outside this download's scratch directory.")
        if temp_dir.exists():
            shutil.rmtree(temp_dir)

    try:
        logger.info(f"正在clone仓库: {git_url} (commit: {commit_sha[:8]}...)")
        
        # 方法1: 尝试浅克隆（更快）
        # 先尝试只clone最近的历史，然后fetch指定commit
        clone_cmd = [
            'git', 'clone', '--depth', '100', '--no-single-branch',
            git_url, str(temp_dir)
        ]
        
        result = subprocess.run(
            clone_cmd,
            capture_output=True,
            text=True,
            timeout=300,
            env=os.environ.copy()  # 使用当前环境变量（包括代理）
        )
        
        if result.returncode == 0:
            # 尝试fetch指定commit
            fetch_cmd = ['git', 'fetch', '--depth', '100', 'origin', commit_sha]
            subprocess.run(
                fetch_cmd,
                cwd=temp_dir,
                capture_output=True,
                timeout=120
            )
            
            # checkout到指定commit
            checkout_cmd = ['git', 'checkout', commit_sha]
            checkout_result = subprocess.run(
                checkout_cmd,
                cwd=temp_dir,
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if checkout_result.returncode != 0:
                # 如果浅克隆失败，尝试完整clone
                logger.info(f"浅克隆中找不到commit，尝试完整clone...")
                clear_checkout()
                result = None
            else:
                logger.info(f"浅克隆成功并checkout到指定commit")
        
        # 方法2: 如果浅克隆失败，使用完整clone
        if result is None or result.returncode != 0:
            if temp_dir.exists():
                clear_checkout()
            
            logger.info(f"使用完整clone方式")
            clone_cmd_full = ['git', 'clone', git_url, str(temp_dir)]
            result_full = subprocess.run(
                clone_cmd_full,
                capture_output=True,
                text=True,
                timeout=600,
                env=os.environ.copy()
            )
            
            if result_full.returncode != 0:
                logger.error(f"完整clone失败: {result_full.stderr}")
                return False
            
            # checkout到指定commit
            checkout_cmd = ['git', 'checkout', commit_sha]
            checkout_result = subprocess.run(
                checkout_cmd,
                cwd=temp_dir,
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if checkout_result.returncode != 0:
                logger.error(f"checkout失败: {checkout_result.stderr}")
                return False
        
        # 使用git archive创建zip文件
        logger.info(f"正在创建zip文件: {zip_file}")
        # 使用-o参数（更通用的格式）
        archive_cmd = ['git', 'archive', '--format=zip', '-o', str(zip_file), commit_sha]
        archive_result = subprocess.run(
            archive_cmd,
            cwd=temp_dir,
            capture_output=True,
            text=True,
            timeout=120
        )
        
        if archive_result.returncode != 0:
            logger.error(f"git archive失败: {archive_result.stderr}")
            return False
        
        # 检查zip文件是否创建成功
        if zip_file.exists() and zip_file.stat().st_size > 0:
            file_size_mb = zip_file.stat().st_size // 1024 // 1024
            logger.info(f"✓ zip文件创建成功: {zip_file.name} ({file_size_mb}MB)")
            return True
        else:
            logger.error(f"zip文件创建失败或文件为空")
            return False
            
    except subprocess.TimeoutExpired:
        logger.error(f"操作超时: {git_url} (commit: {commit_sha[:8]}...)")
        return False
    except Exception as e:
        logger.error(f"创建zip文件时出错: {git_url} (commit: {commit_sha[:8]}...), 错误: {str(e)}")
        return False
    finally:
        scratch.cleanup()


def sanitize_dir_name(name: str) -> str:
    """
    清理目录名，移除或替换不允许的字符
    
    Args:
        name: 原始目录名
        
    Returns:
        str: 清理后的目录名
    """
    # 替换不允许的文件系统字符
    # Windows/Linux不允许的字符: / \ : * ? " < > |
    name = re.sub(r'[<>:"/\\|?*]', '_', name)
    # 移除前后空格和点
    name = name.strip(' .')
    # 限制长度（避免文件名过长）
    if len(name) > 200:
        name = name[:200]
    return name


def generate_pair_dir_name(repo: str, commit_sha: str, sha_length: int = 8) -> str:
    """
    生成pair目录名: repo名字_commit_sha前n位
    
    Args:
        repo: 仓库名
        commit_sha: commit SHA
        sha_length: commit SHA使用的长度（默认8位）
        
    Returns:
        str: 目录名
    """
    repo_clean = sanitize_dir_name(repo)
    sha_prefix = commit_sha[:sha_length] if commit_sha else ''
    dir_name = f"{repo_clean}_{sha_prefix}"
    return dir_name


def save_commit_message(message: str, message_file: Path) -> bool:
    """
    保存commit message到文件
    
    Args:
        message: commit message内容
        message_file: 保存的文件路径
        
    Returns:
        bool: 是否保存成功
    """
    try:
        message_file.parent.mkdir(parents=True, exist_ok=True)
        with open(message_file, 'w', encoding='utf-8') as f:
            f.write(message)
        return True
    except Exception as e:
        logger.error(f"保存commit message失败: {message_file}, 错误: {str(e)}")
        return False


def process_entry(entry: Dict, index: int, output_base_dir: Path, sha_length: int = 8, total: Optional[int] = None) -> Dict:
    """
    处理单个entry，下载仓库和保存message
    
    Args:
        entry: JSON entry数据
        index: entry索引
        output_base_dir: 输出基础目录
        sha_length: commit SHA使用的长度（默认8位）
        
    Returns:
        Dict: 处理结果统计
    """
    progress = f"{index + 1}/{total}" if total is not None else str(index + 1)
    # 使用repo名字和commit_sha前n位生成目录名
    repo_name = entry.get('repo', '')
    commit_sha = entry.get('commit_sha', '')
    owner = entry.get('owner', '')
    pair_dir_name = generate_pair_dir_name(repo_name, commit_sha, sha_length)
    pair_dir = output_base_dir / pair_dir_name
    zip_file = pair_dir / "repo.zip"
    message_file = pair_dir / "commit_message.txt"
    
    result = {
        'index': index,
        'pair_dir_name': pair_dir_name,
        'git_url': entry.get('git_url', ''),
        'commit_sha': commit_sha,
        'repo_downloaded': False,
        'message_saved': False,
        'success': False
    }
    
    # 仅当 repo.zip（非空）与 commit_message.txt 两者都存在时才跳过；仅创建了空目录或缺任一项则重新下载
    zip_ok = zip_file.exists() and zip_file.stat().st_size > 0
    msg_ok = message_file.exists()
    if zip_ok and msg_ok:
        logger.info(f"[{progress}] ⊙ 跳过（已存在）: {entry.get('repo', '')} @ {entry.get('commit_sha', '')[:8]}")
        result['repo_downloaded'] = True
        result['message_saved'] = True
        result['success'] = True
        return result
    if pair_dir.exists() and (not zip_ok or not msg_ok):
        missing = []
        if not zip_ok:
            missing.append("repo.zip" if not zip_file.exists() else "repo.zip(空)")
        if not msg_ok:
            missing.append("commit_message.txt")
        logger.info(f"[{progress}] 目录已存在但缺少 {', '.join(missing)}，将重新下载: {entry.get('repo', '')} @ {entry.get('commit_sha', '')[:8]}")
    
    # 下载仓库zip文件
    repo_success = download_repository_zip(
        owner,
        repo_name,
        commit_sha,
        zip_file
    )
    result['repo_downloaded'] = repo_success
    
    # 保存commit message
    message_success = save_commit_message(
        entry.get('message', ''),
        message_file
    )
    result['message_saved'] = message_success
    
    # 只有两者都成功才算成功
    result['success'] = repo_success and message_success
    
    if result['success']:
        logger.info(f"[{progress}] ✓ 成功处理: {entry.get('repo', '')} @ {entry.get('commit_sha', '')[:8]}")
    else:
        logger.warning(f"[{progress}] ✗ 处理失败: {entry.get('repo', '')} @ {entry.get('commit_sha', '')[:8]} "
                     f"(repo: {repo_success}, message: {message_success})")
    
    return result


def main():
    parser = argparse.ArgumentParser(description="Download repository snapshots and commit messages from a JSONL dataset.")
    parser.add_argument("--input", type=Path, default=PACKAGE_ROOT / "dataset" / "ablation" / "commits" / "python.jsonl")
    parser.add_argument("--output-dir", type=Path, default=PACKAGE_ROOT / "outputs" / "view2" / "repositories")
    parser.add_argument("--verify-token", action="store_true", help="Check GITHUB_TOKEN authentication and exit.")
    args = parser.parse_args()
    if args.verify_token:
        configure_logging()
        raise SystemExit(0 if verify_github_token(get_github_token()) else 1)

    jsonl_file = args.input
    output_base_dir = args.output_dir.resolve()
    if not jsonl_file.is_file():
        parser.error(f"Input file not found: {jsonl_file}")
    with jsonl_file.open("r", encoding="utf-8") as source:
        total = sum(bool(line.strip()) for line in source)
    configure_logging(output_base_dir.parent)
    output_base_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Output directory: %s", output_base_dir)

    stats = {"total": 0, "success": 0, "failed": 0, "repo_download_failed": 0, "message_save_failed": 0}
    start_time = time.time()
    with jsonl_file.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            index = stats["total"]
            stats["total"] += 1
            try:
                entry = json.loads(line)
                if not isinstance(entry, dict):
                    raise ValueError("Expected a JSON object")
                result = process_entry(entry, index, output_base_dir, sha_length=8, total=total)
                if result["success"]:
                    stats["success"] += 1
                else:
                    stats["failed"] += 1
                    stats["repo_download_failed"] += not result["repo_downloaded"]
                    stats["message_save_failed"] += not result["message_saved"]
            except (ValueError, TypeError, KeyError) as error:
                logger.error("Invalid entry on line %s: %s", line_number, error)
                stats["failed"] += 1
            except Exception as error:
                logger.error("Processing line %s failed: %s", line_number, error)
                stats["failed"] += 1
            if stats["total"] % 100 == 0:
                elapsed = time.time() - start_time
                rate = stats["total"] / elapsed if elapsed > 0 else 0
                remaining = (total - stats["total"]) / rate if rate > 0 else 0
                logger.info("Progress: %s/%s; succeeded=%s, failed=%s, remaining=%.1f min", stats["total"], total, stats["success"], stats["failed"], remaining / 60)
    logger.info("Finished: %s; elapsed %.1f min", stats, (time.time() - start_time) / 60)


if __name__ == "__main__":
    main()
