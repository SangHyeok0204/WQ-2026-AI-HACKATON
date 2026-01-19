import re
import matplotlib.pyplot as plt
import networkx as nx

# 각 연산자에 대응하는 함수 매핑
ops = {
    '^': 'power',
    '*': 'multiply',
    '/': 'divide',
    '+': 'add',
    '-': 'subtract',
    '<': 'less',
    '>': 'greater',
    '==': 'equal',
    '&&': 'and',
    '||': 'or'
}

# 함수 정의
def parse_expression(expression):
    def parse(tokens):
        def parse_primary():
            token = tokens.pop(0)
            if token == '(':
                expr = parse_expression_from_tokens(tokens)
                tokens.pop(0)  # for closing parenthesis ')'
                return expr
            elif re.match(r'[a-zA-Z_]\w*\(', token):  # function name with '('
                func_name = token[:-1]
                args = parse_arguments()
                return f'{func_name}({",".join(args)})'
            else:
                return token

        def parse_arguments():
            args = []
            while tokens and tokens[0] != ')':
                if tokens[0] == ',':
                    tokens.pop(0)  # consume the comma
                else:
                    arg = parse_expression_from_tokens(tokens)
                    if tokens and tokens[0] == '=':
                        tokens.pop(0)  # consume the '='
                        value = parse_expression_from_tokens(tokens)
                        arg = f'{arg}={value}'
                    args.append(arg)
            tokens.pop(0)  # for closing parenthesis ')'
            return args
        
        def parse_unary():
            token = tokens[0]
            if token == '-':
                tokens.pop(0)
                return f'reverse({parse_unary()})'
            else:
                return parse_primary()
                
        def parse_exponent():
            expr = parse_unary()
            while tokens and tokens[0] == '^':
                op = tokens.pop(0)
                expr = f'{ops[op]}({expr},{parse_unary()})'
            return expr
            
        def parse_factor():
            expr = parse_exponent()
            while tokens and tokens[0] in '*/':
                op = tokens.pop(0)
                expr = f'{ops[op]}({expr},{parse_exponent()})'
            return expr
            
        def parse_term():
            expr = parse_factor()
            while tokens and tokens[0] in '+-':
                op = tokens.pop(0)
                expr = f'{ops[op]}({expr},{parse_factor()})'
            return expr
            
        def parse_comparative():
            expr = parse_term()
            while tokens and tokens[0] in ['<', '>', '==']:
                op = tokens.pop(0)
                expr = f'{ops[op]}({expr},{parse_term()})'
            return expr
            
        def parse_logical_and():
            expr = parse_comparative()
            while tokens and tokens[0] == '&&':
                op = tokens.pop(0)
                expr = f'{ops[op]}({expr},{parse_comparative()})'
            return expr
        
        def parse_logical_or():
            expr = parse_logical_and()
            while tokens and tokens[0] == '||':
                op = tokens.pop(0)
                expr = f'{ops[op]}({expr},{parse_logical_and()})'
            return expr

        def parse_ternary():
            expr = parse_logical_or()
            if tokens and tokens[0] == '?':
                tokens.pop(0)  # consume the '?'
                true_expr = parse_expression_from_tokens(tokens)
                if tokens and tokens[0] == ':':
                    tokens.pop(0)  # consume the ':'
                    false_expr = parse_expression_from_tokens(tokens)
                    expr = f'if_else({expr},{true_expr},{false_expr})'
            return expr
        
        return parse_ternary()

    def parse_expression_from_tokens(tokens):
        return parse(tokens)

    # 토큰화
    tokens = re.findall(r'\?|\:|[a-zA-Z_]\w*\(|[[a-zA-Z_]\w*|==|&&|\|\||[-+*/()<>=,]|"(?:\\.[^"\\]*)*"|\'(?:\\.[^\'\\]*)*\'|[0-9]*\.?[0-9]+', expression)
    return parse_expression_from_tokens(tokens)

def compress_code(code):
    code = code.replace(' ', '').replace('\n', '').replace('\r','')  # remove spaces and newlines
    pattern = r"(\w+)\s*=\s*(.*?);"
    
    code_line = code.split(';')
    while '' in code_line:
        code_line.remove('')
        
    matches = re.findall(pattern, code)
    
    # create a dictionary with variable names as keys and their values as values
    var_dict = {var: value for var, value in matches}
    
    # replace variable names in the last line of the code with their values
    last_line = code_line[-1]
    
    def replace_vars(last_line, var_dict):
        for var, value in var_dict.items():
            # Use word boundary in regex to ensure whole words are replaced
            last_line = re.sub(r'\b' + re.escape(var) + r'\b', f'({value})', last_line)
        return last_line
        
    attempts = 0
    while last_line != replace_vars(last_line, var_dict):
        attempts += 1
        last_line = replace_vars(last_line, var_dict)
        if attempts >= 100:
            raise Exception(f'compress fail, last_line: "{last_line}"')
            
    return last_line

class TreeNode:
    def __init__(self, value, node_id):
        self.value = value
        self.node_id = node_id
        self.children = []
        self.node_type = None  # 노드 타입 추가

    def calculate_depth(self):
        # If the node has no children, the depth is 1 (itself)
        if not self.children:
            return 0
        # Recursively calculate the depth of all children and add 1 (for the current node)
        return 1 + max(child.calculate_depth() for child in self.children)

    def node_info(self):
        return {
            'node_id': self.node_id,
            'value': self.value,
            'depth': self.calculate_depth(),
            'node_type': self.node_type
        }

    def collect_all_nodes(self, visited=None):
        """중복 노드 제거를 포함한 트리 순회"""
        if visited is None:
            visited = set()  # 이미 방문한 노드 ID를 저장

        nodes = []
        if self.node_id not in visited:   # 아직 방문하지 않은 노드라면 추가
            visited.add(self.node_id)
            nodes.append(self)
            for child in self.children:
                nodes.extend(child.collect_all_nodes(visited))
        return nodes

    def get_children_node_info(self):
        all_nodes = self.collect_all_nodes()
        for x in all_nodes:
            classify_node(x)
        nodes_info = [x.node_info() for x in all_nodes]
        for nodeinfo in nodes_info:
            nodeinfo['reverse_depth'] = max([x['depth'] for x in nodes_info]) - nodeinfo['depth']
        return nodes_info
    
    def node_types(self):
        for x in self.collect_all_nodes():
            classify_node(x)


'''def build_tree(expression):
    def helper(expr):
        stack = []
        root = None
        current_node = None
        node_id = 0  # 노드 ID 초기화
        i = 0
        
        while i < len(expr):
            if expr[i].isalpha() or expr[i] == '_':
                j = i
                while i < len(expr) and (expr[i].isalpha() or expr[i] == '_' or expr[i].isdigit()):
                    i += 1
                    
                if i < len(expr) and expr[i] == '=':
                    k = i + 1
                    if expr[k] == '"':
                        k += 1
                        while k < len(expr) and expr[k] != '"':
                            k += 1
                        k += 1  # include the closing quote
                    else:
                        while k < len(expr) and (expr[k].isalnum() or expr[k] in ('_', '.')):
                            k += 1
                            
                    node = TreeNode(expr[j:k], node_id)
                    node.node_type = "special_argument"
                    node_id += 1
                    if stack:
                        stack[-1].children.append(node)
                    i = k
                else:
                    node = TreeNode(expr[j:i], node_id)
                    node.node_type = "datafield"
                    node_id += 1
                    if not stack:
                        root = node
                    else:
                        stack[-1].children.append(node)
                current_node = node
                continue
            elif expr[i].isdigit() or (expr[i] == '-' and (i + 1 < len(expr) and expr[i + 1].isdigit())):
                j = i
                if expr[i] == '-':
                    i += 1
                while i < len(expr) and (expr[i].isdigit() or expr[i] == '.'):
                    i += 1
                node = TreeNode(expr[j:i], node_id)
                node.node_type = "number"
                node_id += 1
                if stack:
                    stack[-1].children.append(node)
                continue
                
            elif expr[i] == '"':
                j = i
                i += 1
                while i < len(expr) and expr[i] != '"':
                    i += 1
                i += 1  # Include the closing quote
                node = TreeNode(expr[j:i], node_id)
                node.node_type = "datafield"
                node_id += 1
                if stack:
                    stack[-1].children.append(node)
                continue
                
            elif expr[i] == '(':
                stack.append(current_node)
            elif expr[i] == ')':
                stack.pop()
            i += 1
        return root
        
    return helper(expression)'''

def build_tree(expression):
    expression = expression.lower()

    def helper(expr):
        stack = []
        root = None
        current_node = None
        node_id = 0
        i = 0

        def consume_special_argument(expr, name_start, eq_index, base_depth):
            # name_start: 키의 시작 인덱스, eq_index: '=' 위치, base_depth: 현재 괄호 깊이(len(stack))
            k = eq_index + 1
            depth = base_depth
            in_quote = None  # "'", "\"" 중 하나 또는 None

            while k < len(expr):
                ch = expr[k]
                if in_quote:
                    # 따옴표 내부
                    if ch == '\\\\' and k + 1 < len(expr):
                        k += 2  # 이스케이프 문자 건너뛰기
                        continue
                    elif ch == in_quote:
                        in_quote = None
                        k += 1
                        continue
                    else:
                        k += 1
                        continue
                else:
                    # 따옴표 밖
                    if ch == "'" or ch == "\"":
                        in_quote = ch
                        k += 1
                        continue
                    elif ch == '(':
                        depth += 1
                        k += 1
                        continue
                    elif ch == ')':
                        # 현재 인자의 괄호 깊이가 끝나는 ')'면 종료
                        if depth == base_depth:
                            break
                        else:
                            depth -= 1
                            k += 1
                            continue
                    elif ch == ',' and depth == base_depth:
                        # 같은 괄호 깊이의 콤마에서 인자 종료
                        break
                    else:
                        k += 1
                        continue

            return k  # 값의 끝 인덱스(구분자 직전)

        while i < len(expr):
            if expr[i].isalpha() or expr[i] == '_':
                j = i
                while i < len(expr) and (expr[i].isalpha() or expr[i] == '_' or expr[i].isdigit()):
                    i += 1

                # special_argument: name=...
                if i < len(expr) and expr[i] == '=' and not (i + 1 < len(expr) and expr[i + 1] == '='):
                    end = consume_special_argument(expr, j, i, len(stack))
                    node = TreeNode(expr[j:end], node_id)
                    node.node_type = "special_argument"
                    node_id += 1
                    if stack:
                        stack[-1].children.append(node)
                    else:
                        if root is None:
                            root = node
                        else:
                            root.children.append(node)
                    i = end
                    continue
                else:
                    # 일반 식별자(함수명 포함)
                    node = TreeNode(expr[j:i], node_id)
                    node.node_type = "datafield"
                    node_id += 1
                    if not stack:
                        if root is None:
                            root = node
                        else:
                            root = node
                    else:
                        stack[-1].children.append(node)
                    current_node = node
                    # 이후 '('에서 스택에 올려 인자들을 자식으로 연결
                    continue

            elif expr[i].isdigit() or (expr[i] == '-' and (i + 1 < len(expr) and expr[i + 1].isdigit())):
                # 숫자(음수 포함)
                j = i
                if expr[i] == '-':
                    i += 1
                while i < len(expr) and (expr[i].isdigit() or expr[i] == '.'):
                    i += 1
                node = TreeNode(expr[j:i], node_id)
                node.node_type = "number"
                node_id += 1
                if stack:
                    stack[-1].children.append(node)
                else:
                    if root is None:
                        root = node
                continue

            elif expr[i] == "'" or expr[i] == "\"":
                # 문자열 리터럴(이스케이프 처리)
                quote = expr[i]
                j = i
                i += 1
                while i < len(expr):
                    if expr[i] == '\\\\' and i + 1 < len(expr):
                        i += 2
                        continue
                    if expr[i] == quote:
                        i += 1
                        break
                    i += 1
                node = TreeNode(expr[j:i], node_id)
                node.node_type = "datafield"
                node_id += 1
                if stack:
                    stack[-1].children.append(node)
                else:
                    if root is None:
                        root = node
                continue

            elif expr[i] == '(':
                # 현재 노드를 부모로 하여 인자들을 연결
                stack.append(current_node)
                i += 1
                continue

            elif expr[i] == ')':
                if stack:
                    stack.pop()
                i += 1
                continue

            else:
                # 기타 문자(콤마 등) 스킵
                i += 1

        return root

    return helper(expression)

def tree_node(exp):
    node = build_tree(parse_expression(exp))
    [classify_node(x) for x in node.collect_all_nodes()]
    return node

def classify_node(node):
    if node.children:
        node.node_type = "operator"
    elif '=' in node.value:
        node.node_type = "special_argument"
    elif node.value.replace('.', '', 1).isdigit() or (node.value[1:].replace('.', '', 1).isdigit() and node.value[0] == '-'):
        node.node_type = "number"
    else:
        node.node_type = "datafield"    

def add_edges(graph, node, parent=None):
    classify_node(node)  # 노드 타입 결정
    graph.add_node(node.node_id, label=node.value, node_type=node.node_type)
    if parent is not None:
        graph.add_edge(parent.node_id, node.node_id, weight=1)
    for child in node.children:
        add_edges(graph, child, node)

def get_node_color(node_type):
    colors = {
        "operator": "skyblue",
        "special_argument": "orange",
        "number": "green",
        "datafield": "pink"
    }
    return colors.get(node_type, "gray")

def hierarchy_pos(G, root, width=1., vert_gap=0.2, vert_loc=0, xcenter=0.5):
    pos = _hierarchy_pos(G, root, width, vert_gap, vert_loc, xcenter)
    return pos

def _hierarchy_pos(G, root, width=1., vert_gap=0.2, vert_loc=0, xcenter=0.5, pos=None, parent=None, parsed=[]):
    if pos is None:
        pos = {root: (xcenter, vert_loc)}
    else:
        pos[root] = (xcenter, vert_loc)
        
    children = list(G.neighbors(root))
    if not isinstance(G, nx.DiGraph) and parent is not None:
        children.remove(parent)
    if len(children) != 0:
        dx = width / len(children)
        nextx = xcenter - width/2 - dx/2
        for child in children:
            nextx += dx
            pos = _hierarchy_pos(G, child, width=dx, vert_gap=vert_gap, vert_loc=vert_loc-vert_gap, xcenter=nextx, pos=pos, parent=root, parsed=parsed)
    return pos

def draw_tree(root):
    graph = nx.Graph()  # 무방향 그래프 사용
    add_edges(graph, root)
    
    pos = hierarchy_pos(graph, root.node_id)
    labels = nx.get_node_attributes(graph, 'label')
    node_types = nx.get_node_attributes(graph, 'node_type')
    colors = [get_node_color(node_types[node]) for node in graph.nodes]
    
    plt.figure(figsize=(10, 8))
    nx.draw(graph, pos, with_labels=True, labels=labels, node_color=colors, node_size=3000, font_size=10, font_weight="bold", arrows=False)
    plt.title("Tree Visualization", size=15)
    plt.show()
    
    return graph

def make_network(root):
    graph = nx.Graph()  # 무방향 그래프 사용
    add_edges(graph, root)
    return graph

def draw_graph(graph):
    root_id = list(graph.nodes)[0]  # 그래프의 첫 번째 노드를 루트 노드로 가정
    pos = hierarchy_pos(graph, root_id)
    labels = nx.get_node_attributes(graph, 'label')
    node_types = nx.get_node_attributes(graph, 'node_type')
    colors = [get_node_color(node_types[node]) for node in graph.nodes]
    
    plt.figure(figsize=(10, 8))
    nx.draw(graph, pos, with_labels=True, labels=labels, node_color=colors, node_size=3000, font_size=10, font_weight="bold", arrows=False)
    plt.title("Tree Visualization", size=15)
    plt.show()

def find_datafield_nodes(graph):
    datafield_nodes = []
    for node_id, data in graph.nodes(data=True):
        if data['node_type'] == "datafield":
            datafield_nodes.append(node_id)
    return datafield_nodes

def calculate_distance(graph, node1_id, node2_id):
    try:
        distance = nx.shortest_path_length(graph, source=node1_id, target=node2_id, weight='weight')
    except nx.NetworkXNoPath:
        return float('inf')
    return distance

def print_datafield_distances(graph, datafield_nodes):
    datafield_distances = []
    n = len(datafield_nodes)
    for i in range(n):
        for j in range(i + 1, n):
            node1_id = datafield_nodes[i]
            node2_id = datafield_nodes[j]
            label1 = graph.nodes[node1_id]['label']
            label2 = graph.nodes[node2_id]['label']
            distance = calculate_distance(graph, node1_id, node2_id)
            print(f"Distance between '{label1}' and '{label2}': {distance}")
            datafield_distances.append([label1, label2, distance])
    return datafield_distances

def datafield_distances(graph):
    datafield_nodes = find_datafield_nodes(graph)
    datafield_distances = []
    n = len(datafield_nodes)
    for i in range(n):
        for j in range(i + 1, n):
            node1_id = datafield_nodes[i]
            node2_id = datafield_nodes[j]
            label1 = graph.nodes[node1_id]['label']
            label2 = graph.nodes[node2_id]['label']
            distance = calculate_distance(graph, node1_id, node2_id)
            datafield_distances.append([label1, label2, distance])
    return datafield_distances

def get_operators_from_datafield(graph, start_node_no):
    operators = []
    root_id = list(graph.nodes)[0]  # 그래프의 첫 번째 노드를 루트 노드로 가정
    path = nx.shortest_path(graph, source=list(graph.nodes)[start_node_no], target=root_id)
    for node_id in path:
        if graph.nodes[node_id]['node_type'] == 'operator':
            operators.append(graph.nodes[node_id]['label'])
    return operators

def get_operators_distance_from_datafield(graph, start_node_no):
    operators = []
    root_id = list(graph.nodes)[0]  # 그래프의 첫 번째 노드를 루트 노드로 가정
    path = nx.shortest_path(graph, source=list(graph.nodes)[start_node_no], target=root_id)
    for node_id in path:
        if graph.nodes[node_id]['node_type'] == 'operator':
            operators.append((graph.nodes[node_id]['label'], nx.shortest_path_length(graph, start_node_no, node_id)))
    return operators

def flatten_list_twice(nested_list):
    result = []
    for item in nested_list:
        if isinstance(item, list):
            result.extend(item)
        else:
            result.append(item)
    return result

def get_match_datafield(G, df_name):
    return [(x[0], x[1]) for x in sorted(dict(G[df_name]).items(), key=lambda x: x[1]['weight'], reverse=True) if G.nodes[x[0]]['type']=='datafield']

def get_match_operator(G, df_name):
    return [(x[0], x[1]) for x in sorted(dict(G[df_name]).items(), key=lambda x: x[1]['weight'], reverse=True) if G.nodes[x[0]]['type']=='operator']