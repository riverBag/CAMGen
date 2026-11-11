from tree_sitter import Language, Parser
import os
import tree_sitter_python as tspython
import tree_sitter_java as tsjava

# 设置Tree-sitter
def setup_tree_sitter():
    # Use installed package
    PY_LANGUAGE = Language(tspython.language())
    parser = Parser(PY_LANGUAGE)
    return parser

def setup_java_parser():
    JAVA_LANGUAGE = Language(tsjava.language())
    parser = Parser(JAVA_LANGUAGE)
    return parser

def treeSitter_file(parser, file_path):
    with open(file_path, 'rb') as f:
        source_code = f.read()

    # 解析代码
    tree = parser.parse(source_code)
    return tree, source_code

def get_node_at_line(root_node, line_number):
    """
    Find the smallest named node that covers the given line number (1-based).
    """
    target_line = line_number - 1 # Convert to 0-based
    
    # Use named_descendant_for_point_range to find the node.
    # We don't have exact columns, so we can try spanning the whole line?
    # Or stick to the recursive search which is safe.
    # Let's try to improve the recursive search to drill down correctly.
    
    current = root_node
    while True:
        # Check children
        found_child = False
        for child in current.children:
            # Check if child fully contains the target line OR overlaps with it significantly?
            # Usually we want the smallest node that *contains* the point of interest.
            # But we only have line number.
            # If a child starts <= target_line and ends >= target_line, it's a candidate.
            if child.start_point[0] <= target_line and child.end_point[0] >= target_line:
                # But we want to avoid picking a huge block if there is a smaller one inside.
                # Example: Block (5-10) contains Loop (6-9). Target 7.
                # Both contain 7. We want Loop.
                # So we drill down.
                current = child
                found_child = True
                break
        if not found_child:
            break
            
    return current

def get_minimal_logical_block(parser, file_path, line_number):
    """
    Returns the start and end line (1-based) of the minimal logical block
    surrounding the given line number.
    
    Logical block logic:
    - Nearest control flow statement (if, try, for, while, switch, etc.)
    - If none, the statement itself (expression_statement, variable_declaration)
    - If top level in method, the statement.
    """
    try:
        tree, source_code = treeSitter_file(parser, file_path)
        root_node = tree.root_node
        
        node = get_node_at_line(root_node, line_number)
        if node is None:
             return line_number, line_number
        
        # Traverse up
        current = node
        candidate_statement = None
        
        # Interesting control flow types (Java)
        control_flow_types = {
            'if_statement',
            'try_statement',
            'try_with_resources_statement',
            'for_statement',
            'enhanced_for_statement',
            'while_statement',
            'do_statement',
            'switch_expression',
            'switch_statement',
            'synchronized_statement',
        }
        
        statement_types = {
            'expression_statement',
            'local_variable_declaration',
            'return_statement',
            'throw_statement',
            'assert_statement',
            'break_statement',
            'continue_statement',
            'yield_statement',
            'declaration'
        }
        
        boundary_types = {
            'method_declaration',
            'constructor_declaration',
            'class_declaration',
            'interface_declaration',
            'enum_declaration',
            'class_body', 
        }
        
        result_node = None
        
        # Debug: print ancestry
        # temp = node
        # while temp:
        #     print(f"Ancestry: {temp.type} ({temp.start_point[0]+1}-{temp.end_point[0]+1})")
        #     temp = temp.parent
        
        while current:
            node_type = current.type
            
            # If we hit a control flow statement, this is our winner.
            if node_type in control_flow_types:
                result_node = current
                break
                
            # If we hit a statement, record it as a fallback, but keep looking for control flow parents.
            # Check if the node is effectively a statement type or wraps one
            if node_type in statement_types:
                if candidate_statement is None:
                    candidate_statement = current
            
            # Check if it is a block
            if node_type == 'block':
                # A block usually belongs to a control flow (if, for) or a method.
                # If we are in a block, and haven't found a statement, maybe the line is a comment or something.
                # Continue up.
                pass

            # If we hit a boundary, we stop searching up.
            if node_type in boundary_types:
                # If we found a statement on the way up, return that.
                if candidate_statement:
                    result_node = candidate_statement
                else:
                    # If no statement found.
                    # Special case: The line might be the start of a statement that we missed because we started too deep?
                    # Or maybe it's a 'local_variable_declaration' that is not in our set? (It is).
                    
                    # Check if the node itself is covering the line.
                    result_node = node
                break
                
            current = current.parent
            
        if not result_node:
            # If we ran out of parents (root), use candidate or node
            result_node = candidate_statement if candidate_statement else node
            
        # If the result node is 'local_variable_declaration', it should have been caught by 'statement_types'.
        # If we returned just 'node', it means we hit boundary without finding 'statement_types'.
        
        # If the found node is very small (like just an identifier), and we want the full line statement:
        # We might need to check if the node is part of a larger structure that starts on the same line?
        # But get_node_at_line tries to find the deepest node.
        
        # Let's try to expand result_node if it's not a block/statement.
        if result_node.type not in control_flow_types and result_node.type not in statement_types:
            # Try to find a parent that is a statement or control flow
            temp = result_node
            while temp and temp.type not in boundary_types:
                if temp.type in statement_types or temp.type in control_flow_types:
                    result_node = temp
                    break
                temp = temp.parent

        if result_node:
             start = result_node.start_point[0] + 1
             end = result_node.end_point[0] + 1
             # Ensure the found block actually covers the requested line
             if line_number < start or line_number > end:
                 return line_number, line_number
             return start, end
        else:
             return line_number, line_number

    except Exception:
        return line_number, line_number



def parse_file(parser, file_path):
    # 解析文件
    tree, source_code = treeSitter_file(parser, file_path)
    
    # 获取根节点
    root_node = tree.root_node

    return root_node, source_code

def print_node(node, source_code, level=0):
    # 打印节点类型和文本
    node_text = source_code[node.start_byte:node.end_byte].decode('utf-8')
    if len(node_text) > 50:
        node_text = node_text[:47] + "..."
    print('  ' * level + f"{node.type}: '{node_text}'")
    
    # 递归打印子节点
    for child in node.children:
        print_node(child, source_code, level + 1)

def analyze_imports(root_node, source_code):
    imports = []
    
    # 查找所有import语句
    import_nodes = root_node.children
    for node in import_nodes:
        if node.type == "import_statement" or node.type == "import_from_statement":
            imports.append(source_code[node.start_byte:node.end_byte].decode('utf-8'))
    
    return imports

def analyze_functions(root_node, source_code):
    functions = []
    
    # 查找所有函数定义
    for node in root_node.children:
        if node.type == "function_definition":
            # 获取函数名和起始行号
            for child in node.children:
                if child.type == "identifier":
                    func_name = source_code[child.start_byte:child.end_byte].decode('utf-8')
                    # 获取函数的起始行号和结束行号（行号从0开始，所以+1使其从1开始）
                    start_line = node.start_point[0] + 1
                    end_line = node.end_point[0] + 1
                    functions.append((func_name, start_line, end_line))
    
    return functions

def analyze_classes(root_node, source_code):
    classes = {}
    
    # 查找所有类定义
    for node in root_node.children:
        if node.type == "class_definition":
            class_name = None
            methods = []
            
            # 获取类的起始行号和结束行号
            class_start_line = node.start_point[0] + 1
            class_end_line = node.end_point[0] + 1
            
            # 获取类名
            for child in node.children:
                if child.type == "identifier":
                    class_name = source_code[child.start_byte:child.end_byte].decode('utf-8')
                
                # 查找类中的方法
                if child.type == "block":
                    for block_child in child.children:
                        # 处理普通方法定义
                        if block_child.type == "function_definition":
                            method_name = None
                            for func_child in block_child.children:
                                if func_child.type == "identifier":
                                    method_name = source_code[func_child.start_byte:func_child.end_byte].decode('utf-8')
                                    # 获取方法的起始行号和结束行号
                                    start_line = block_child.start_point[0] + 1
                                    end_line = block_child.end_point[0] + 1
                                    methods.append((method_name, start_line, end_line))
                                    break
                        
                        # 处理带装饰器的方法定义
                        elif block_child.type == "decorated_definition":
                            for decorated_child in block_child.children:
                                if decorated_child.type == "function_definition":
                                    method_name = None
                                    for func_child in decorated_child.children:
                                        if func_child.type == "identifier":
                                            method_name = source_code[func_child.start_byte:func_child.end_byte].decode('utf-8')
                                            # 获取方法的起始行号和结束行号
                                            start_line = block_child.start_point[0] + 1  # 使用装饰器的起始行
                                            end_line = decorated_child.end_point[0] + 1
                                            methods.append((method_name, start_line, end_line))
                                            break
            
            if class_name:
                # 存储类名、起始行号、结束行号和方法列表
                classes[class_name] = (class_start_line, class_end_line, methods)
    
    return classes

# 修改获取方法函数以适应新的数据结构
def get_class_methods(class_name, classes_dict):
    if class_name in classes_dict:
        return classes_dict[class_name][2]  # 返回方法列表
    else:
        return []

def save_analysis_to_file(file_path, classes, functions, imports, output_dir):
    """将解析结果按行号顺序保存到JSON文件中，使方法嵌套在类中
    
    Args:
        file_path: 源文件路径
        classes: 类分析结果
        functions: 函数分析结果
        imports: 导入语句分析结果
        output_dir: 输出目录路径，如果为None则使用默认目录
    
    Returns:
        输出文件的路径
    """
    import json
    
    # 创建元素列表
    elements = []
    
    # 添加类及其方法（嵌套结构）
    for cls_name, (start_line, end_line, methods) in classes.items():
        # 转换方法列表格式
        methods_list = []
        for method_name, method_start, method_end in methods:
            methods_list.append({
                'name': method_name,
                'start_line': method_start,
                'end_line': method_end
            })
        
        # 创建类对象，包含嵌套的方法列表
        class_obj = {
            'type': 'class',
            'name': cls_name,
            'start_line': start_line,
            'end_line': end_line,
            'methods': methods_list
        }
        elements.append(class_obj)
    
    # 添加独立的函数
    for func_name, start_line, end_line in functions:
        elements.append({
            'type': 'function',
            'name': func_name,
            'start_line': start_line,
            'end_line': end_line
        })
    
    # 按照起始行号排序
    elements.sort(key=lambda x: x['start_line'])
    
    # 创建JSON数据结构
    analysis_data = {
        'file_name': os.path.basename(file_path),
        'file_path': os.path.abspath(file_path),  # 添加文件的完整路径
        'imports': imports,
        'elements': elements
    }
    
    
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 创建基于源文件路径的文件名
    # 将路径中的特殊字符替换为下划线
    file_name = file_path.replace(':', '_').replace('\\', '_').replace('/', '_')
    output_path = os.path.join(output_dir, f"{file_name}.json")
    
    # 写入JSON文件
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(analysis_data, f, ensure_ascii=False, indent=2)
        
    print(f"\n分析结果已保存到: {output_path}")
    return output_path



def parse_file_and_save(file_path,output_dir):
    """解析文件并保存为JSON格式"""
    try:
        # 设置Tree-sitter
        parser = setup_tree_sitter()
        
        # 解析文件
        root_node, source_code = parse_file(parser, file_path)
        
        # 分析导入、函数和类
        imports = analyze_imports(root_node, source_code)
        functions = analyze_functions(root_node, source_code)
        classes = analyze_classes(root_node, source_code)
        
        # 保存分析结果到JSON文件
        output_path = save_analysis_to_file(file_path, classes, functions, imports,output_dir)
        
        return output_path
    except Exception as e:
        print(f"处理文件 {file_path} 时出错: {str(e)}")
        return None

def process_file(file_path, output_dir):
    """集成函数：解析指定文件并生成相应的的json分析文件
    
    Args:
        file_path: 要分析的Python文件路径
        output_dir: 输出目录路径，如果为None则使用默认目录
        
    Returns:
        生成的json文件路径
    """
    try:
        print(f"\n开始解析文件: {file_path}")
        
        # 调用parse_file_and_save函数
        output_path = parse_file_and_save(file_path, output_dir)
        
        if output_path:
            # 输出简要分析结果
            print(f"\n文件 {os.path.basename(file_path)} 已成功解析")
            print(f"分析结果已保存到: {output_path}")
        
        return output_path
    
    except Exception as e:
        print(f"处理文件 {file_path} 时出错: {str(e)}")
        return None
