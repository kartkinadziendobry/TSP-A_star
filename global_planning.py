"""

A* grid planning with multi-point inspection path planning

"""

import math
import itertools
import matplotlib.pyplot as plt
import numpy as np
# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题


# 只在最后显示结果图像，不显示中间过程动画
show_animation = False

# 是否显示最终结果图像
show_final_plot = True


class AStarPlanner:

    def __init__(self, ox, oy, resolution, rr):
        """
        Initialize grid map for a star planning

        ox: x position list of Obstacles [m]
        oy: y position list of Obstacles [m]
        resolution: grid resolution [m],地图的像素
        rr: robot radius[m]
        """

        self.resolution = resolution
        self.rr = rr
        self.min_x, self.min_y = 0, 0
        self.max_x, self.max_y = 0, 0
        self.obstacle_map = None
        self.x_width, self.y_width = 0, 0
        self.motion = self.get_motion_model()
        # 确保障碍物列表不为空，否则无法确定地图边界
        if not ox or not oy:
            # 提供一个默认的最小地图范围，如果ox, oy为空
            ox_default = [0, 1]
            oy_default = [0, 1]
            self.calc_obstacle_map(ox_default, oy_default)
            # 清空障碍物图，因为初始障碍物是临时的
            self.obstacle_map = [[False for _ in range(self.y_width)]
                                 for _ in range(self.x_width)]
            if ox and oy: # 如果用户之后又传入了障碍物，则重新计算
                 self.calc_obstacle_map(ox, oy)
        else:
            self.calc_obstacle_map(ox, oy)


    class Node:
        """定义搜索区域节点类,每个Node都包含坐标x和y, 移动代价cost和父节点索引。
        """
        def __init__(self, x, y, cost, parent_index):
            self.x = x  # index of grid
            self.y = y  # index of grid
            self.cost = cost
            self.parent_index = parent_index

        def __str__(self):
            return str(self.x) + "," + str(self.y) + "," + str(
                self.cost) + "," + str(self.parent_index)

    def planning(self, sx, sy, gx, gy):
        """
        A star path search
        输入起始点和目标点的坐标(sx,sy)和(gx,gy)，
        最终输出的结果是路径包含的点的坐标集合rx和ry。
        input:
            s_x: start x position [m]
            s_y: start y position [m]
            gx: goal x position [m]
            gy: goal y position [m]

        output:
            rx: x position list of the final path
            ry: y position list of the final path
        """

        start_node = self.Node(self.calc_xy_index(sx, self.min_x),
                               self.calc_xy_index(sy, self.min_y), 0.0, -1)
        goal_node = self.Node(self.calc_xy_index(gx, self.min_x),
                              self.calc_xy_index(gy, self.min_y), 0.0, -1)

        open_set, closed_set = dict(), dict()
        open_set[self.calc_grid_index(start_node)] = start_node

        # 设置最大迭代次数，防止陷入死循环
        max_iterations = 100000 # 增加迭代次数以应对更复杂的场景
        iterations = 0

        while 1:
            iterations += 1
            if len(open_set) == 0 or iterations > max_iterations:
                if len(open_set) == 0:
                    print(f"无法找到从({sx:.2f}, {sy:.2f})到({gx:.2f}, {gy:.2f})的路径，将尝试使用直线连接。")
                else:
                    print(f"迭代次数({iterations})超过限制({max_iterations})，终止搜索，将尝试使用直线连接从({sx:.2f}, {sy:.2f})到({gx:.2f}, {gy:.2f})。")
                
                # 如果无法找到路径，返回起点和终点的直线连接
                # 同时检查直线是否穿过障碍物 (简化检查，仅检查中点)
                direct_rx, direct_ry = [sx, gx], [sy, gy]
                mid_x, mid_y = (sx+gx)/2, (sy+gy)/2
                mid_node = self.Node(self.calc_xy_index(mid_x, self.min_x), self.calc_xy_index(mid_y, self.min_y), 0, -1)
                if not self.verify_node(mid_node): # 检查中点是否在障碍物内
                     print(f"警告: 从({sx:.2f}, {sy:.2f})到({gx:.2f}, {gy:.2f})的直线路径可能穿过障碍物。")
                return direct_rx, direct_ry


            c_id = min(
                open_set,
                key=lambda o: open_set[o].cost + self.calc_heuristic(goal_node, open_set[o]))
            current = open_set[c_id]

            # show graph
            if show_animation:  # pragma: no cover
                plt.plot(self.calc_grid_position(current.x, self.min_x),
                         self.calc_grid_position(current.y, self.min_y), "xc")
                # for stopping simulation with the esc key.
                plt.gcf().canvas.mpl_connect('key_release_event', lambda event: [exit(0) if event.key == 'escape' else None])
                if len(closed_set.keys()) % 10 == 0:
                    plt.pause(0.001)

            if current.x == goal_node.x and current.y == goal_node.y:
                # print(f"成功找到路径: 从({sx:.2f}, {sy:.2f})到({gx:.2f}, {gy:.2f})")
                goal_node.parent_index = current.parent_index
                goal_node.cost = current.cost
                break

            del open_set[c_id]
            closed_set[c_id] = current

            for i, _ in enumerate(self.motion):
                node = self.Node(current.x + self.motion[i][0],
                                 current.y + self.motion[i][1],
                                 current.cost + self.motion[i][2], c_id)
                n_id = self.calc_grid_index(node)

                if not self.verify_node(node):
                    continue
                if n_id in closed_set:
                    continue
                if n_id not in open_set:
                    open_set[n_id] = node
                else:
                    if open_set[n_id].cost > node.cost:
                        open_set[n_id] = node

        rx, ry = self.calc_final_path(goal_node, closed_set)
        return rx, ry

    def calc_final_path(self, goal_node, closed_set):
        rx, ry = [self.calc_grid_position(goal_node.x, self.min_x)], [
            self.calc_grid_position(goal_node.y, self.min_y)]
        parent_index = goal_node.parent_index
        while parent_index != -1:
            n = closed_set[parent_index]
            rx.append(self.calc_grid_position(n.x, self.min_x))
            ry.append(self.calc_grid_position(n.y, self.min_y))
            parent_index = n.parent_index
        return rx[::-1], ry[::-1]

    @staticmethod
    def calc_heuristic(n1, n2):
        w = 1.0
        d = w * math.hypot(n1.x - n2.x, n1.y - n2.y)
        return d

    def calc_grid_position(self, index, min_position):
        pos = index * self.resolution + min_position
        return pos

    def calc_xy_index(self, position, min_pos):
        return round((position - min_pos) / self.resolution)

    def calc_grid_index(self, node):
        # 检查node.x和node.y是否在有效范围内
        if not (self.min_x <= node.x < self.min_x + self.x_width and
                self.min_y <= node.y < self.min_y + self.y_width):
             # 如果节点在 calc_xy_index 转换后就在地图边界之外，则索引可能无效
             # 例如，如果 min_x 是 0, calc_xy_index(-1, 0) -> -1, 这是无效的
             # 这种情况通常由 verify_node 处理，但这里多一层保护
             # 返回一个特殊值或抛出异常，或者确保 verify_node 先被调用
             pass # 依赖 verify_node
        
        # 确保索引基于0开始的地图内部索引
        # 即 node.x 应该是相对于 self.min_x / self.resolution 的偏移量
        # 但 Node 存储的是栅格索引，不是实际坐标
        # calc_grid_index 期望的是栅格索引
        
        # 确保node.x和node.y是相对于地图左下角的索引
        # 当前 Node 中的 x, y 已经是栅格索引了
        return (node.y - 0) * self.x_width + (node.x - 0) # 假设 Node.x, Node.y 是从0开始的索引
        # return (node.y - self.calc_xy_index(self.min_y, self.min_y)) * self.x_width + \
        #        (node.x - self.calc_xy_index(self.min_x, self.min_x))


    def verify_node(self, node):
        # node.x 和 node.y 是栅格索引
        # 需要检查这些索引是否在 self.obstacle_map 的边界内
        if not (0 <= node.x < self.x_width and 0 <= node.y < self.y_width):
            return False # 超出地图栅格范围

        # collision check using obstacle_map (indexed by grid indices)
        if self.obstacle_map[int(node.x)][int(node.y)]: # 确保索引是整数
            return False
        return True

    def calc_obstacle_map(self, ox, oy):
        if not ox or not oy: # 如果没有障碍物，ox, oy可能是空列表
            # 设置一个默认的最小地图，例如从0到1，防止min/max出错
            self.min_x = 0
            self.min_y = 0
            self.max_x = self.resolution # 至少一个格子的宽度
            self.max_y = self.resolution # 至少一个格子的高度
        else:
            self.min_x = round(min(ox))
            self.min_y = round(min(oy))
            self.max_x = round(max(ox))
            self.max_y = round(max(oy))

        # print("min_x:", self.min_x)
        # print("min_y:", self.min_y)
        # print("max_x:", self.max_x)
        # print("max_y:", self.max_y)

        # 确保max_x > min_x and max_y > min_y
        if self.max_x <= self.min_x: self.max_x = self.min_x + self.resolution
        if self.max_y <= self.min_y: self.max_y = self.min_y + self.resolution


        self.x_width = round((self.max_x - self.min_x) / self.resolution)
        self.y_width = round((self.max_y - self.min_y) / self.resolution)
        
        # 确保宽度至少为1
        if self.x_width <= 0: self.x_width = 1
        if self.y_width <= 0: self.y_width = 1

        # print("x_width:", self.x_width)
        # print("y_width:", self.y_width)

        self.obstacle_map = [[False for _ in range(self.y_width)]
                             for _ in range(self.x_width)]
        
        if not ox or not oy: # 如果没有障碍物，则直接返回空的障碍物图
            return

        for ix in range(self.x_width):
            map_x_coord = self.calc_grid_position(ix, self.min_x) # ix是栅格索引
            for iy in range(self.y_width):
                map_y_coord = self.calc_grid_position(iy, self.min_y) # iy是栅格索引
                for obs_x, obs_y in zip(ox, oy):
                    d = math.hypot(obs_x - map_x_coord, obs_y - map_y_coord)
                    if d <= self.rr:
                        self.obstacle_map[ix][iy] = True
                        break

    @staticmethod
    def get_motion_model():
        motion = [[1, 0, 1], [0, 1, 1], [-1, 0, 1], [0, -1, 1],
                  [-1, -1, math.sqrt(2)], [-1, 1, math.sqrt(2)],
                  [1, -1, math.sqrt(2)], [1, 1, math.sqrt(2)]]
        return motion

    def plan_path_between_points(self, sx, sy, gx, gy):
        rx, ry = self.planning(sx, sy, gx, gy)
        path_length = 0
        if len(rx) >= 2:
            for i in range(len(rx) - 1):
                path_length += math.hypot(rx[i+1] - rx[i], ry[i+1] - ry[i])
        elif len(rx) == 1 and sx == gx and sy == gy : # 起点终点相同
             pass # 长度为0
        else: # 路径规划失败，使用直线长度
            print(f"路径规划失败或路径点不足，计算({sx:.2f},{sy:.2f})到({gx:.2f},{gy:.2f})的直线距离。")
            path_length = math.hypot(gx - sx, gy - sy)
            # 如果planning返回的是直线，rx, ry已经是[sx,gx], [sy,gy]
        return rx, ry, path_length

# --- 新增辅助函数 ---
def calculate_turbine_inner_path(turbine_points, a_star_planner):
    # 根据新逻辑，此函数用于表示"到达并完成单个选定点的巡检"
    # 实际路径长度为0，路径只包含该选定点。
    # 然而，为了保持solve_tsp_for_turbines的结构（期望内部距离），
    # 这个函数现在主要用于返回0长度。
    # 在新的TSP逻辑中，我们将直接使用0作为内部成本。
    # 如果确实需要返回选定点的路径（例如，用于非常详细的日志），可以这样做：
    # if turbine_points and len(turbine_points) == 1: # 假设传入的是单个选定点
    #     return [turbine_points[0][0]], [turbine_points[0][1]], 0
    # else: # 或者传入原始的两个点，但我们只关心其"内部成本"为0
    #     print(f"calculate_turbine_inner_path: 期望单个选定点或用于成本计算，返回0成本。实际点数: {len(turbine_points) if turbine_points else 0}")
    return [], [], 0 # 返回空路径和0成本，因为TSP将处理单个点之间的连接

# --- 修改后的TSP求解器，针对风机组 ---
def solve_tsp_for_turbines(global_start_point, chosen_turbine_nodes, a_star_planner): # 参数名修改
    """
    解决TSP问题，确定单点代表的风机组的访问顺序。
    Args:
        global_start_point: (x,y) 全局起点/终点
        chosen_turbine_nodes: [(x1,y1), (x2,y2), ...] 每个风机选定的单个巡检点坐标列表
        a_star_planner: AStarPlanner对象
    Returns:
        optimal_turbine_order: 最优风机组访问顺序 (风机组索引列表)
        min_total_distance: 最短总距离 (只包含机组间路径，内部路径为0)
        all_inter_turbine_paths_dict: 字典，存储机组间连接的路径 {(from_idx, to_idx): (rx, ry)}
                                     from_idx/to_idx: -1代表全局起点, 0到n-1代表风机组索引
    """
    num_turbines = len(chosen_turbine_nodes)
    if num_turbines == 0:
        rx, ry, dist = a_star_planner.plan_path_between_points(global_start_point[0], global_start_point[1], global_start_point[0], global_start_point[1])
        return [], dist, {(-1,-1):(rx,ry)}

    distances = {}
    paths_data = {}

    # 1. 全局起点到每个风机的选定巡检点的路径
    for i in range(num_turbines):
        start_x, start_y = global_start_point
        target_node_x, target_node_y = chosen_turbine_nodes[i]
        rx, ry, dist = a_star_planner.plan_path_between_points(start_x, start_y, target_node_x, target_node_y)
        distances[(-1, i)] = dist
        paths_data[(-1, i)] = (rx, ry, dist)

    # 2. 每个风机的选定巡检点到其他风机的选定巡检点的路径
    for i in range(num_turbines):
        from_node_x, from_node_y = chosen_turbine_nodes[i]
        for j in range(num_turbines):
            if i == j:
                continue
            to_node_x, to_node_y = chosen_turbine_nodes[j]
            rx, ry, dist = a_star_planner.plan_path_between_points(
                from_node_x, from_node_y,
                to_node_x, to_node_y
            )
            distances[(i, j)] = dist
            paths_data[(i, j)] = (rx, ry, dist)

    # 3. 每个风机的选定巡检点回到全局起点的路径
    for i in range(num_turbines):
        from_node_x, from_node_y = chosen_turbine_nodes[i]
        end_x, end_y = global_start_point
        rx, ry, dist = a_star_planner.plan_path_between_points(from_node_x, from_node_y, end_x, end_y)
        distances[(i, -1)] = dist
        paths_data[(i, -1)] = (rx, ry, dist)
        
    # 4. 风机内部路径长度为0
    turbine_internal_distances = [0.0] * num_turbines


    # TSP 求解 (基于排列)
    turbine_indices = list(range(num_turbines))
    all_permutations = list(itertools.permutations(turbine_indices))

    min_total_distance = float('inf')
    optimal_turbine_order_indices = None # 存储风机组的索引顺序

    if not all_permutations and num_turbines > 0: # 处理只有一个风机的情况
        all_permutations = [tuple(turbine_indices)]
    elif num_turbines == 0:
        optimal_turbine_order_indices = []
        min_total_distance = distances.get((-1,-1), 0)


    for perm_indices in all_permutations:
        current_total_distance = 0
        
        # 路径: global_start -> perm_indices[0] -> perm_indices[1] -> ... -> perm_indices[-1] -> global_start
        
        # 1. 从全局起点到第一个风机组
        first_turbine_idx = perm_indices[0]
        current_total_distance += distances[(-1, first_turbine_idx)]
        current_total_distance += turbine_internal_distances[first_turbine_idx] # 加上第一个风机的内部路径

        # 2. 风机组之间的路径
        for k in range(len(perm_indices) - 1):
            from_turbine_idx = perm_indices[k]
            to_turbine_idx = perm_indices[k+1]
            current_total_distance += distances[(from_turbine_idx, to_turbine_idx)]
            current_total_distance += turbine_internal_distances[to_turbine_idx] # 加上下一个风机的内部路径

        # 3. 从最后一个风机组回到全局起点
        last_turbine_idx = perm_indices[-1]
        current_total_distance += distances[(last_turbine_idx, -1)]

        if current_total_distance < min_total_distance:
            min_total_distance = current_total_distance
            optimal_turbine_order_indices = list(perm_indices)
            
    # 如果只有一个风机组
    if num_turbines == 1 and not optimal_turbine_order_indices:
        optimal_turbine_order_indices = [0]
        min_total_distance = distances[(-1,0)] + turbine_internal_distances[0] + distances[(0,-1)]


    # 提取TSP决策的连接路径 (机组与机组之间，起点到机组，机组到终点)
    all_inter_turbine_paths_dict = {}
    for key, val in paths_data.items():
        all_inter_turbine_paths_dict[key] = (val[0], val[1]) # rx, ry

    return optimal_turbine_order_indices, min_total_distance, all_inter_turbine_paths_dict


# --- 修改后的多点路径规划函数 ---
def multi_point_path_planning_hierarchical(ox, oy, global_start_point, chosen_turbine_nodes, original_turbines_inspect_points, grid_size=2.0, robot_radius=1.0): # 参数名和新增参数
    """
    分层多目标点路径规划: TSP确定风机组顺序 (基于选定单点), A*规划段路径。
    Args:
        ox, oy: 障碍物坐标列表
        global_start_point: (x,y) 全局起点/终点
        chosen_turbine_nodes: [(x1,y1), (x2,y2), ...] 每个风机选定的单个巡检点坐标列表
        original_turbines_inspect_points: [[(P0x,P0y),(P1x,P1y)],...] 原始定义的两点，用于绘图
        grid_size, robot_radius: A*参数
    Returns:
        full_rx, full_ry: 完整巡检路径的x, y坐标列表
        optimal_turbine_order_indices: 风机组的最优访问顺序 (索引列表)
        total_path_length_check: 校验用的总路径长度
    """
    a_star = AStarPlanner(ox, oy, grid_size, robot_radius)
    num_turbines = len(chosen_turbine_nodes)

    print(f"开始分层路径规划: 全局起点 {global_start_point}, 风机数量 {num_turbines} (每个风机一个选定巡检点)")

    if num_turbines == 0:
        print("没有风机组需要巡检。")
        rx_final, ry_final, _ = a_star.plan_path_between_points(global_start_point[0], global_start_point[1], global_start_point[0], global_start_point[1])
        return rx_final, ry_final, [], 0

    # 1. 求解风机组的TSP访问顺序 (基于选定的单个巡检点)
    optimal_turbine_order_indices, tsp_min_dist, _ = solve_tsp_for_turbines( # inter_paths_dict 可能不需要了，因为连接点固定
        global_start_point, chosen_turbine_nodes, a_star
    )
    
    if optimal_turbine_order_indices is None :
        print("错误:未能解析TSP获取风机顺序")
        optimal_turbine_order_indices = list(range(num_turbines))


    print(f"TSP计算完成。最优风机访问顺序 (索引): {optimal_turbine_order_indices}, 估计总距离: {tsp_min_dist:.2f}")
    
    # 2. 构建完整路径
    full_rx, full_ry = [], []
    calculated_total_length = 0

    current_pos = global_start_point
    full_rx.append(current_pos[0])
    full_ry.append(current_pos[1])

    # 2.1 路径: 全局起点 -> 第一个风机的选定点
    if num_turbines > 0:
        first_turbine_idx = optimal_turbine_order_indices[0]
        target_point = chosen_turbine_nodes[first_turbine_idx] # 直接使用选定点
        
        print(f"规划路径: 全局起点 {current_pos} -> 风机 {first_turbine_idx} 的选定点 {target_point}")
        rx_seg, ry_seg, len_seg = a_star.plan_path_between_points(current_pos[0], current_pos[1], target_point[0], target_point[1])
        calculated_total_length += len_seg
        full_rx.extend(rx_seg[1:])
        full_ry.extend(ry_seg[1:])
        current_pos = (full_rx[-1], full_ry[-1])

    # 2.2 遍历排序后的风机组
    for i in range(len(optimal_turbine_order_indices)):
        turbine_idx = optimal_turbine_order_indices[i]
        # 当前风机的选定巡检点 (也是此风机巡检的起点和终点)
        current_turbine_selected_node = chosen_turbine_nodes[turbine_idx]
        
        print(f"\n处理风机 {turbine_idx} (顺序中的第 {i+1} 个), 巡检点: {current_turbine_selected_node}")

        # 确保当前位置 current_pos 就是 current_turbine_selected_node
        # (上一步的终点应该是这一步的起点)
        if not (abs(current_pos[0] - current_turbine_selected_node[0]) < 1e-3 and abs(current_pos[1] - current_turbine_selected_node[1]) < 1e-3):
            print(f"警告: 当前位置 {current_pos} 与风机 {turbine_idx} 的选定点 {current_turbine_selected_node} 不匹配。可能路径有误。")
            # 这里应该已经是匹配的，因为上一步的 target_point 就是 current_turbine_selected_node
            # 并且 current_pos 更新为该路径的终点

        # 内部路径成本为0，不实际移动，只更新 current_pos 为此风机的选定点 (实际上它已经是了)
        # calculated_total_length += 0 (由TSP的 turbine_internal_distances 控制)
        # full_rx, full_ry 也不需要添加点，因为我们已经到达了这个点，并将从这个点出发

        # 2.2.2 路径: 当前风机的选定点 -> 下一个风机的选定点
        if i < len(optimal_turbine_order_indices) - 1:
            next_turbine_idx = optimal_turbine_order_indices[i+1]
            target_next_point = chosen_turbine_nodes[next_turbine_idx] # 下一个风机的选定点
            
            print(f"规划连接路径: 风机 {turbine_idx} (从选定点 {current_pos}) -> 风机 {next_turbine_idx} 的选定点 {target_next_point}")
            rx_seg, ry_seg, len_seg = a_star.plan_path_between_points(current_pos[0], current_pos[1], target_next_point[0], target_next_point[1])
            calculated_total_length += len_seg
            full_rx.extend(rx_seg[1:])
            full_ry.extend(ry_seg[1:])
            current_pos = (full_rx[-1], full_ry[-1])
    
    # 2.3 路径: 最后一个风机的选定点 -> 全局起点
    if num_turbines > 0:
        print(f"规划返回路径: 最后一个风机 (当前在 {current_pos}) -> 全局起点 {global_start_point}")
        rx_seg, ry_seg, len_seg = a_star.plan_path_between_points(current_pos[0], current_pos[1], global_start_point[0], global_start_point[1])
        calculated_total_length += len_seg
        full_rx.extend(rx_seg[1:])
        full_ry.extend(ry_seg[1:])
        current_pos = (full_rx[-1], full_ry[-1]) # 应回到全局起点

    print(f"\n完整路径构建完成。总计算长度: {calculated_total_length:.2f}")
    print(f"路径起点: ({full_rx[0]:.2f}, {full_ry[0]:.2f}), 路径终点: ({full_rx[-1]:.2f}, {full_ry[-1]:.2f})")
    
    # 检查路径是否闭环
    if not (abs(full_rx[0] - full_rx[-1]) < 0.1 and abs(full_ry[0] - full_ry[-1]) < 0.1):
        print(f"警告: 最终路径不是严格闭环。起点({full_rx[0]:.2f}, {full_ry[0]:.2f}), 终点({full_rx[-1]:.2f}, {full_ry[-1]:.2f})")
        # 可以选择强制闭环
        # full_rx.append(full_rx[0])
        # full_ry.append(full_ry[0])
        # print("已强制添加闭环点。")


    return full_rx, full_ry, optimal_turbine_order_indices, calculated_total_length


# --- Bezier Curve Smoothing Functions ---
def _cubic_bezier_point(p0, p1, p2, p3, t):
    """计算三次贝塞尔曲线在参数t处的一个点"""
    x0, y0 = p0
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3

    inv_t = 1.0 - t
    inv_t_sq = inv_t * inv_t
    inv_t_cub = inv_t_sq * inv_t

    t_sq = t * t
    t_cub = t_sq * t

    x = (inv_t_cub * x0 +
         3 * inv_t_sq * t * x1 +
         3 * inv_t * t_sq * x2 +
         t_cub * x3)
    y = (inv_t_cub * y0 +
         3 * inv_t_sq * t * y1 +
         3 * inv_t * t_sq * y2 +
         t_cub * y3)
    return (x, y)

def smooth_path_with_bezier(path_xy, tangent_scale_factor=1.0/3.0, num_samples_per_segment=10):
    """
    使用三次贝塞尔曲线平滑路径。

    Args:
        path_xy: 包含(x,y)坐标元组的列表。
        tangent_scale_factor: 用于缩放切线以确定控制点的因子。
                              默认1.0/3.0与Catmull-Rom到Bezier转换相关。
        num_samples_per_segment: 每个原始路径段生成的平滑点数。

    Returns:
        包含(x,y)平滑路径点的列表。
    """
    n = len(path_xy)
    if n < 2:
        return path_xy

    smoothed_path_xy = []

    if n == 2:  # 对于只有两点的直线路径
        p0, p1 = path_xy[0], path_xy[1]
        for k_sample in range(num_samples_per_segment + 1):
            t = k_sample / float(num_samples_per_segment)
            x = p0[0] * (1.0 - t) + p1[0] * t
            y = p0[1] * (1.0 - t) + p1[1] * t
            smoothed_path_xy.append((x, y))
        # 确保最后一个点精确匹配
        if smoothed_path_xy and smoothed_path_xy[-1] != p1 :
             # 检查浮点数问题，如果最后一个采样点不是完全等于p1
             dist_sq_to_p1 = (smoothed_path_xy[-1][0] - p1[0])**2 + (smoothed_path_xy[-1][1] - p1[1])**2
             if dist_sq_to_p1 > 1e-9: # 如果相差较大，则替换
                 smoothed_path_xy[-1] = p1
             elif not smoothed_path_xy: # 如果列表为空（理论上不会发生）
                 smoothed_path_xy.append(p1)


        return smoothed_path_xy

    # 路径有3个或更多点
    smoothed_path_xy.append(path_xy[0])  # 添加第一个原始点

    for i in range(n - 1):  # 遍历每个原始路径段 (P_i, P_{i+1})
        p_i = path_xy[i]
        p_i_plus_1 = path_xy[i+1]

        # 计算P_i处的切线向量分量 (m_t_i)
        if i == 0:  # 第一个点
            # 外插 P_{i-1} 以使切线从 P_i 指向 P_{i+1}
            p_i_minus_1 = (2 * p_i[0] - p_i_plus_1[0], 2 * p_i[1] - p_i_plus_1[1])
        else:
            p_i_minus_1 = path_xy[i-1]
        
        m_t_i_x = (p_i_plus_1[0] - p_i_minus_1[0]) / 2.0
        m_t_i_y = (p_i_plus_1[1] - p_i_minus_1[1]) / 2.0

        # 计算P_{i+1}处的切线向量分量 (m_t_i_plus_1)
        if i == n - 2:  # P_{i+1}是最后一个点
            # 外插 P_{i+2} 以使切线从 P_i 指向 P_{i+1}
            p_i_plus_2 = (2 * p_i_plus_1[0] - p_i[0], 2 * p_i_plus_1[1] - p_i[1])
        else:
            p_i_plus_2 = path_xy[i+2]

        m_t_i_plus_1_x = (p_i_plus_2[0] - p_i[0]) / 2.0
        m_t_i_plus_1_y = (p_i_plus_2[1] - p_i[1]) / 2.0

        # 贝塞尔控制点
        b0 = p_i
        b3 = p_i_plus_1

        # 第一个中间控制点 B1
        cp1_x = p_i[0] + m_t_i_x * tangent_scale_factor
        cp1_y = p_i[1] + m_t_i_y * tangent_scale_factor
        b1 = (cp1_x, cp1_y)

        # 第二个中间控制点 B2
        cp2_x = p_i_plus_1[0] - m_t_i_plus_1_x * tangent_scale_factor
        cp2_y = p_i_plus_1[1] - m_t_i_plus_1_y * tangent_scale_factor
        b2 = (cp2_x, cp2_y)

        # 为该段贝塞尔曲线采样点
        # 从 k_sample = 1 开始，因为 smoothed_path_xy 已包含 b0 (即 P_i)
        for k_sample in range(1, num_samples_per_segment + 1):
            t = k_sample / float(num_samples_per_segment)
            point_on_curve = _cubic_bezier_point(b0, b1, b2, b3, t)
            smoothed_path_xy.append(point_on_curve)
            
    return smoothed_path_xy
# --- End of Bezier Curve Smoothing Functions ---

def main():
    print(__file__ + " start!!")

    grid_size = 2.0 # [m]
    robot_radius = 1.5 # [m] # 减小一点，给路径更多空间

    # 定义全局起点/终点
    global_start_point = (5.0, 5.0)

    # 定义障碍物中心 (与原代码类似，选择一种形状)
    ox, oy = [], []
    # 边界
    map_size = 50.0
    border_points = int(map_size / 1.0) # 控制边界点的密度
    ox.extend([i for i in np.linspace(0, map_size, border_points)])
    oy.extend([0.0] * border_points)
    ox.extend([map_size] * border_points)
    oy.extend([i for i in np.linspace(0, map_size, border_points)])
    ox.extend([i for i in np.linspace(map_size, 0, border_points)])
    oy.extend([map_size] * border_points)
    ox.extend([0.0] * border_points)
    oy.extend([i for i in np.linspace(map_size, 0, border_points)])


    # # ============= 形状3：圆形排列的风机障碍物 =============
    # obstacle_centers = []
    # num_obs_turbines = 3 # 作为障碍物的风机数量
    # center_x_obs, center_y_obs = 25, 25
    # radius_obs = 15
    # for i in range(num_obs_turbines):
    #     angle = i * 2 * math.pi / num_obs_turbines
    #     x = center_x_obs + radius_obs * math.cos(angle)
    #     y = center_y_obs + radius_obs * math.sin(angle)
    #     obstacle_centers.append((x,y))
    
    # # 为每个障碍物中心添加实际的障碍物点
    # obstacle_size = 1.0 # 障碍物半径或半边长
    # for center_x, center_y in obstacle_centers:
    #     for dx_obs in np.linspace(-obstacle_size, obstacle_size, 3): # 3x3点云代表障碍物
    #         for dy_obs in np.linspace(-obstacle_size, obstacle_size, 3):
    #             ox.append(center_x + dx_obs)
    #             oy.append(center_y + dy_obs)

    # 定义风机组及其4个巡检点
    # 示例：3个风机组
    turbines_inspect_points = []
    
    # 定义风机位置和创建巡检点
    turbine_centers = [] # 初始化风机中心列表

    # ============= 选择一种风机排列形状 (取消注释以选择) =============
    # 注意：请确保风机数量在3到10个左右，并调整参数以适应地图大小 (0-50)

    # --- 形状1：矩形/正方形排列 ---
    #num_rows = 3
    #num_cols = 3  # 总风机数 = num_rows * num_cols (例如 2x3 = 6 个风机)
    #spacing_x = 15.0
    #spacing_y = 15.0
    #start_x = 10.0
    #start_y = 10.0
    #for r in range(num_rows):
     #    for c in range(num_cols):
    #          x = start_x + c * spacing_x
    #          y = start_y + r * spacing_y
    #          turbine_centers.append((x, y))

    # --- 形状2：圆形排列 ---
    #num_turbines_circle = 8 # 风机数量 (推荐 3-10)
    #center_x_circle, center_y_circle = 25.0, 25.0  # 圆心
    #radius_circle = 15.0  # 圆半径
    #for i in range(num_turbines_circle):
     #  angle = i * 2 * math.pi / num_turbines_circle
     #   x = center_x_circle + radius_circle * math.cos(angle)
     #   y = center_y_circle + radius_circle * math.sin(angle)
     #   turbine_centers.append((x, y))

     #--- 形状3：扇形排
    
    num_turbines_fan_total = 6 # 风机总数量 (推荐 3-10)
    center_x_fan, center_y_fan = 25.0, 25.0  # 扇形顶点
    radius_fan_outer = 20.0  # 外层扇形半径
    radius_fan_inner = 12.0  # 内层扇形半径 (例如 外层半径 * 0.6)
    start_angle_fan = 0.0  # 起始角度 (弧度)
    end_angle_fan = math.pi   # 结束角度 (弧度, 例如 math.pi 表示180度扇形)

    # 分配内外层风机数量
    # 使用 math.ceil 和 math.floor 来分配，确保总数正确，外层优先或均分
    num_turbines_outer_float = 6
    num_turbines_inner_float = 0
    
    num_turbines_outer = int(num_turbines_outer_float)
    num_turbines_inner = int(num_turbines_inner_float)

    # 放置外层风机
    if num_turbines_outer > 0:
        if num_turbines_outer == 1:
            # 如果该层只有一个风机，则角步长为0，风机将位于起始角度
            angle_step_outer = 0.0
        else: # num_turbines_outer > 1
            # 如果 start_angle_fan == end_angle_fan, angle_step_outer 将为0, 所有点重合在起始角度
            angle_step_outer = (end_angle_fan - start_angle_fan) / (num_turbines_outer - 1)
        
        for i in range(num_turbines_outer):
            angle = start_angle_fan + i * angle_step_outer
            x = center_x_fan + radius_fan_outer * math.cos(angle)
            y = center_y_fan + radius_fan_outer * math.sin(angle)
            turbine_centers.append((x, y))

    # 放置内层风机
    if num_turbines_inner > 0:
        if num_turbines_inner == 1:
            # 如果该层只有一个风机，则角步长为0，风机将位于起始角度
            angle_step_inner = 0.0
        else: # num_turbines_inner > 1
            angle_step_inner = (end_angle_fan - start_angle_fan) / (num_turbines_inner - 1)

        for i in range(num_turbines_inner):
            angle = start_angle_fan + i * angle_step_inner
            x = center_x_fan + radius_fan_inner * math.cos(angle)
            y = center_y_fan + radius_fan_inner * math.sin(angle)
            turbine_centers.append((x, y))
    
    # --- 形状4：椭圆形排列 ---
    #num_turbines_ellipse = 10 # 风机数量 (推荐 3-10)
    #center_x_ellipse, center_y_ellipse = 25.0, 25.0  # 椭圆中心
    #a_ellipse, b_ellipse = 20.0, 10.0  # 椭圆长短半轴
    #for i in range(num_turbines_ellipse):
    #    angle = i * 2 * math.pi / num_turbines_ellipse
    #    x = center_x_ellipse + a_ellipse * math.cos(angle)
    #    y = center_y_ellipse + b_ellipse * math.sin(angle)
    #    turbine_centers.append((x, y))

    # --- 形状5：三角形排列 (以顶点为例) ---
    # # 定义大三角形的三个顶点作为风机位置
    #p1 = (10.0, 10.0)
    #p2 = (40.0, 10.0)
    #p3 = (25.0, 35.0)
    #turbine_centers.extend([p1, p2, p3])
    # # 如果需要更多风机，可以考虑在边上或内部添加
    # # 例如，添加三边中点 (总共6个风机)
    #turbine_centers.append(((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2))
    #turbine_centers.append(((p2[0] + p3[0]) / 2, (p2[1] + p3[1]) / 2))
    #turbine_centers.append(((p3[0] + p1[0]) / 2, (p3[1] + p1[1]) / 2))

    #--- 形状6：线形排列 ---
    #num_turbines_line = 6 # 风机数量
    #start_point_line = (5.0, 25.0)
    #end_point_line = (45.0, 25.0)
    #if num_turbines_line > 1:
     #   for i in range(num_turbines_line):
      #      ratio = i / (num_turbines_line - 1)
       #     x = start_point_line[0] * (1 - ratio) + end_point_line[0] * ratio
        #    y = start_point_line[1] * (1 - ratio) + end_point_line[1] * ratio
         #   turbine_centers.append((x,y))
    #elif num_turbines_line == 1: 
     #   turbine_centers.append(start_point_line)


    # 默认风机位置 (如果没有选择以上形状，则使用这个)
    
    if not turbine_centers:
        turbine_centers = [
            (15.0, 15.0),  # 风机1
            (35.0, 15.0),  # 风机2
            (25.0, 35.0),  # 风机3
            (10.0, 35.0),  # 风机4
            (40.0, 40.0)   # 风机5
        ]
        print("警告：未使用特定形状排列风机，已采用默认风机位置。")
    


    d = 3  # 巡检点与中心的距离

    # 为每个风机创建巡检点和障碍物
    for center in turbine_centers:
        # 添加两个巡检点 (例如，中心左右两侧)
        # 顺序: (cx - d, cy), (cx + d, cy)
        turbines_inspect_points.append([
            (center[0] - d, center[1]), # P0
            (center[0] + d, center[1])  # P1
        ])

        # 将风机本身设为障碍物 (3x3点云代表风机塔筒或基础)
        turbine_obstacle_size = 0.5 # 风机作为障碍物的半径或半边长
        for dx_obs in [-turbine_obstacle_size, 0, turbine_obstacle_size]:
            for dy_obs in [-turbine_obstacle_size, 0, turbine_obstacle_size]:
                ox.append(center[0] + dx_obs)
                oy.append(center[1] + dy_obs)
            
    # (可选) 添加一些额外的随机障碍物
    # for _ in range(5):
    #     rand_x = np.random.uniform(10, 40)
    #     rand_y = np.random.uniform(10, 40)
    #     is_far_from_turbines = True
    #     all_turbine_centers = [t1_center, t2_center, t3_center]
    #     for tc in all_turbine_centers:
    #         if math.hypot(rand_x - tc[0], rand_y - tc[1]) < d * 3: # 不要太靠近风机巡检路径
    #             is_far_from_turbines = False
    #             break
    #     if is_far_from_turbines:
    #          for dx_rand in [-1,0,1]:
    #              for dy_rand in [-1,0,1]:
    #                 ox.append(rand_x + dx_rand)
    #                 oy.append(rand_y + dy_rand)


    # 执行分层多目标点路径规划
    print(f"地图边界: min_x={min(ox) if ox else 0}, max_x={max(ox) if ox else 1}, min_y={min(oy) if oy else 0}, max_y={max(oy) if oy else 1}")

    # 为每个风机选择第一个定义的点作为其唯一巡检点
    chosen_turbine_nodes_for_tsp = []
    if turbines_inspect_points:
        for turbine_two_points in turbines_inspect_points:
            if turbine_two_points: # 确保列表不为空
                chosen_turbine_nodes_for_tsp.append(turbine_two_points[0]) # 选择第一个点 P0
            else:
                print("警告: 风机定义点列表为空，跳过此风机。")
    
    print(f"为TSP选择的各风机巡检点: {chosen_turbine_nodes_for_tsp}")


    rx, ry, optimal_turbine_order, total_dist = multi_point_path_planning_hierarchical(
        ox, oy, global_start_point, chosen_turbine_nodes_for_tsp, turbines_inspect_points, grid_size, robot_radius
    )

    # 使用贝塞尔曲线平滑路径
    if rx and ry:
        original_path_xy = list(zip(rx, ry))
        # 可以调整 tangent_scale_factor 和 num_samples_per_segment
        # tangent_scale_factor 较小 -> 更接近原始折线; 较大 -> 可能更圆滑但偏离原始点更远
        # num_samples_per_segment 越大 -> 曲线越精细，点越多
        smoothed_path_xy = smooth_path_with_bezier(original_path_xy, tangent_scale_factor=0.33, num_samples_per_segment=15) 
        
        if smoothed_path_xy:
            smooth_rx = [p[0] for p in smoothed_path_xy]
            smooth_ry = [p[1] for p in smoothed_path_xy]
            print(f"路径已平滑，原始点数: {len(rx)}, 平滑后点数: {len(smooth_rx)}")
        else: # 平滑失败或路径过短
            smooth_rx, smooth_ry = rx, ry # 使用原始路径
            print("路径平滑未执行或失败，使用原始路径。")
    else:
        smooth_rx, smooth_ry = [], []
        print("无路径可平滑。")


    # 绘图
    if show_final_plot:
        plt.figure(figsize=(8, 8))
        # 绘制障碍物
        if ox and oy:
            plt.scatter(ox, oy, marker='s', color='k', s=100, label="风电机组")
        
        # 绘制全局起点/终点
        plt.plot(global_start_point[0], global_start_point[1], "og", markersize=12, label="全局起点/终点")

        # 绘制风机组的巡检点和访问顺序
        
        turbine_colors = plt.cm.viridis(np.linspace(0, 1, len(turbines_inspect_points)))

        for i, turbine_idx in enumerate(optimal_turbine_order):
            # original_turbine_points 用于获取原始定义的两个点进行绘图
            original_turbine_points = turbines_inspect_points[turbine_idx] 
            # chosen_node 是此风机实际被访问的点
            chosen_node = chosen_turbine_nodes_for_tsp[turbine_idx]

            color = turbine_colors[turbine_idx % len(turbine_colors)] # 使用风机原始索引turbine_idx来决定颜色，保持一致性
            
            # 绘制该风机原始定义的2个巡检点
            for j, point in enumerate(original_turbine_points):
                plt.plot(point[0], point[1], "x", color=color, markersize=8)
                # 标记巡检点在风机内部的顺序 (P0, P1)
                plt.text(point[0] + 0.5, point[1] + 0.5, f"T{turbine_idx}.P{j}", fontsize=8, color=color)
            
            # 高亮显示被选作实际巡检的那个点
            #plt.plot(chosen_node[0], chosen_node[1], "o", markersize=10, markerfacecolor='None', markeredgecolor=color, markeredgewidth=2, label=f"风机{turbine_idx}选定点" if i == 0 else None)


            # 绘制两个巡检点间距离为直径的圆形 (基于原始定义的两个点)
            if len(original_turbine_points) >= 2:
                p0, p1 = original_turbine_points[0], original_turbine_points[1]
                distance = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
                # 计算圆心（两点的中点）
                center_x = (p0[0] + p1[0]) / 2
                center_y = (p0[1] + p1[1]) / 2
                # 绘制圆形
                circle = plt.Circle((center_x, center_y), distance/2, fill=False, color='b', linestyle='-', linewidth=1.5)
                plt.gca().add_patch(circle)
            # 标记风机组的TSP访问顺序
            # 取风机组的几何中心近似标记
            turbine_center_x = sum(p[0] for p in original_turbine_points) / len(original_turbine_points) if original_turbine_points else center[0] # 使用len(original_turbine_points)
            turbine_center_y = sum(p[1] for p in original_turbine_points) / len(original_turbine_points) if original_turbine_points else center[1] # 使用len(original_turbine_points)
            plt.text(turbine_center_x, turbine_center_y - d*1.5, f"Order: {i+1}", fontsize=12, color='red', weight='bold', ha='center')

        # 绘制完整巡检路径
        
        if rx and ry: # 绘制原始A*路径作为参考（可选）
            # plt.plot(rx, ry, ":", color='gray', linewidth=1.0, label="原始A*路径")
            pass

        if smooth_rx and smooth_ry:
            plt.plot(smooth_rx, smooth_ry, "-b", linewidth=1.5, label="巡检路径") # 使用蓝色表示平滑路径
            
            # 在平滑路径上添加箭头指示方向
            path_len_for_arrows = len(smooth_rx)
            num_arrows = min(30, path_len_for_arrows // 10 if path_len_for_arrows > 10 else 1) # 调整箭头密度
            if num_arrows > 1 and path_len_for_arrows > 1:
                 arrow_positions = np.linspace(0, path_len_for_arrows - 2, num_arrows, dtype=int)
                 for k_arrow in arrow_positions:
                    plt.arrow(smooth_rx[k_arrow], smooth_ry[k_arrow],
                              (smooth_rx[k_arrow+1] - smooth_rx[k_arrow]) * 0.8, 
                              (smooth_ry[k_arrow+1] - smooth_ry[k_arrow]) * 0.8, # 调整箭头长度比例
                              head_width=0.5, head_length=0.5, # 调整箭头大小
                              fc='blue', ec='blue', length_includes_head=True)
        elif rx and ry: # 如果平滑失败，则绘制原始路径
            plt.plot(rx, ry, "-r", linewidth=1.5, label="巡检路径 (原始)")


        #plt.title(f"分层风电场巡检路径规划 (TSP+A*)\n风机访问顺序 (索引): {optimal_turbine_order}\n总路径长度: {total_dist:.2f} m")
        plt.xlabel("X [n mile]")
        plt.ylabel("Y [n mile]")
        plt.grid(True)
        plt.axis("equal")
        plt.legend(loc='upper right', fontsize='small')
        
        plt.savefig('hierarchical_path_planning_result.png', dpi=300)
        print("路径规划结果图像已保存为 hierarchical_path_planning_result.png")
        plt.show()

if __name__ == '__main__':
    main()

