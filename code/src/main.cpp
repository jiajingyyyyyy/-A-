/**
 * @/mnt/c/Users/86153/Desktop/人工智能导论/A_star
 * 地图1 ： 228 * 152
 * 地图2 ：118 * 96
 * 默认模式为自动 + 地图1， 切换注释可切换梯度，SINGLE_STEP_MODE置为1可切换单步运行
*/
#include<iostream>
#include<cmath>
#include<limits>
#include<queue>
#include<vector>
#include<opencv2/opencv.hpp>
#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#define SINGLE_STEP_MODE 1 //一个字符迭代一次

using namespace std;


class Node{
public:
  int x;
  int y;
  float sum_cost;
  Node* p_node;

  Node(int x_, int y_, float sum_cost_=0, Node* p_node_=NULL):x(x_), y(y_), sum_cost(sum_cost_), p_node(p_node_){};
};

// 定义栅格大小和分辨率
const int gridWidth = 200;    // 栅格宽度
const int gridHeight = 200;   // 栅格高度

// 定义栅格数据结构
typedef std::vector<std::vector<int>> GridData;

// 将图像转换为栅格数据结构
GridData convertImageToGrid(const cv::Mat& image) {
    GridData grid(image.rows, std::vector<int>(image.cols));

    for (int i = 0; i < image.rows; ++i) {
        for (int j = 0; j < image.cols; ++j) {
            // 获取像素值
            int pixelValue = image.at<uchar>(i, j);
            // 判断像素是否为黑色（障碍物）
            if(pixelValue >= 127) grid[i][j] = 0;
            else grid[i][j] = 1;
            // if(pixelValue >= 127) grid[i][j] = 1;
            // else grid[i][j] = 0;
        }
    }

    return grid;
}
std::vector<std::vector<int>> create_map_from_grid(const std::vector<std::vector<int>>& grid, float reso, cv::Mat& img, int img_reso) {
  // 获取栅格地图的宽度和高度
  int width = grid[0].size();
  int height = grid.size();
  //cout << "width in createmap" << width<<endl;
  // 创建地图，初始化为全0
  std::vector<std::vector<int>> map(height, std::vector<int>(width, 0));

  // 遍历栅格地图的每个网格
  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      // 如果栅格值为1，则表示该网格为障碍物
      if (grid[i][j] == 1) {
        map[i][j] = 1;
        // 在图像img上标记障碍物，将对应的图像网格填充为黑色
        cv::rectangle(img,
                      cv::Point(j * img_reso + 1, i * img_reso + 1),
                      cv::Point((j + 1) * img_reso, (i + 1) * img_reso),
                      cv::Scalar(0, 0, 0), -1);
      }
    }
  }

  // 返回创建的地图
  return map;
}
std::vector<std::vector<float> > calc_final_path(Node * goal, float reso, cv::Mat& img, float img_reso){
  std::vector<float> rx;
  std::vector<float> ry;// 存储路径的x , y坐标
  Node* node = goal;
  int total_dis = 0;
  // 从目标节点开始回溯路径，直到达到起始节点（起始节点的父节点为NULL）
  while (node->p_node != NULL){
    node = node->p_node;
	// 将离散化的路径坐标（乘以分辨率reso）添加到rx和ry向量中
    rx.push_back(node->x * reso);
    ry.push_back(node->y * reso);
	// 在图像img上标记路径，将路径对应的图像网格标记为蓝色
    cv::rectangle(img,
        cv::Point(node->x*img_reso+1, node->y*img_reso+1),
        cv::Point((node->x+1)*img_reso, (node->y+1)*img_reso),
        cv::Scalar(255, 0, 0), -1);
        total_dis ++;
  }
  // 返回包含路径离散坐标的二维浮点数向量
  cout << "the total dis under this heuristic function is ";
  cout << total_dis << endl;
  return {rx, ry};
  
}


std::vector<std::vector<int>> calc_obstacle_map(
    std::vector<int> ox, std::vector<int> oy,
    const int min_ox, const int max_ox,
    const int min_oy, const int max_oy,
    float reso, float vr,
    cv::Mat& img, int img_reso) {

  int xwidth = max_ox - min_ox;  // 障碍物地图的宽度
  int ywidth = max_oy - min_oy;  // 障碍物地图的高度
  //cout << "width in calc_obstacle_map" << xwidth <<endl;
  // 创建障碍物地图，初始化为全0
  std::vector<std::vector<int>> obmap(ywidth, std::vector<int>(xwidth, 0));

  //遍历障碍物地图的每个网格
  for (int i = 0; i < xwidth; i++) {
    int x = i + min_ox;  // 当前网格的x坐标
    for (int j = 0; j < ywidth; j++) {
      int y = j + min_oy;  // 当前网格的y坐标
      for (int k = 0; k < ox.size(); k++) {
        // 计算当前网格与障碍物之间的距离
        float d = std::sqrt(std::pow((ox[k] - x), 2) + std::pow((oy[k] - y), 2));
        if (d <= vr / reso) {
          // 如果距离小于等于虚拟半径vr/reso，则将该网格标记为障碍物
          obmap[i][j] = 1;
          // 在图像img上标记障碍物，将对应的图像网格填充为黑色
          cv::rectangle(img,
                        cv::Point(i * img_reso + 1, j * img_reso + 1),
                        cv::Point((i + 1) * img_reso, (j + 1) * img_reso),
                        cv::Scalar(0, 0, 0), -1);
          break;
        }
      }
    }
  }

  // 返回障碍物地图
  return obmap;
}

/**
 *检验点是否合法
*/
bool verify_node(Node* node,
                 const vector<vector<int>>& obmap,
                 int min_ox, int max_ox,
                 int min_oy, int max_oy){
  if (node->x < min_ox || node->y < min_oy || node->x >= max_ox || node->y >= max_oy){
    return false;
  }

  if (obmap[node->x-min_ox][node->y-min_oy]) return false;

  return true;
}


/**
 * 计算启发式函数（Heuristic function）的值，用于估计两个节点之间的代价（距离）。
 *
 * @param n1 第一个节点
 * @param n2 第二个节点
 * @param w 权重参数，默认为1.0
 * @return 两个节点之间的启发式函数值（代价）
 */

float calc_heuristic(Node* n1, Node* n2, float w = 1.0) {
  // 根据节点的坐标计算欧几里得距离，并乘以权重参数w
  return w * std::sqrt(std::pow(n1->x - n2->x, 2) + std::pow(n1->y - n2->y, 2));
}

// /**
//  * 计算曼哈顿距离作为启发式函数的值，用于估计两个节点之间的代价。
//  *
//  * @param n1 第一个节点
//  * @param n2 第二个节点
//  * @param w 权重参数，默认为1.0
//  * @return 两个节点之间的曼哈顿距离启发式函数值（代价）
//  */
// float calc_heuristic(Node* n1, Node* n2, float w = 1.0) {
//   // 计算节点之间的水平和垂直距离
//   float dx = std::abs(n1->x - n2->x);
//   float dy = std::abs(n1->y - n2->y);
//   // 乘以权重参数w并返回启发式函数值（代价）
//   return w * (dx + dy);
// }

// /**
//  * 计算切比雪夫距离作为启发式函数的值，用于估计两个节点之间的代价。
//  *
//  * @param n1 第一个节点
//  * @param n2 第二个节点
//  * @param w 权重参数，默认为1.0
//  * @return 两个节点之间的切比雪夫距离启发式函数值（代价）
//  */
// float calc_heuristic(Node* n1, Node* n2, float w = 1.0) {
//   // 计算节点之间的水平和垂直距离
//   float dx = std::abs(n1->x - n2->x);
//   float dy = std::abs(n1->y - n2->y);
//   // 取水平和垂直距离的最大值，乘以权重参数w并返回启发式函数值（代价）
//   return w * std::max(dx, dy);
// }
//定义了机器人在规划过程中可以移动的方向和代价
std::vector<Node> get_motion_model(){
  return {Node(1,   0,  1),
          Node(0,   1,  1),
          Node(-1,   0,  1),
          Node(0,   -1,  1),
          Node(-1,   -1,  std::sqrt(2)),
          Node(-1,   1,  std::sqrt(2)),
          Node(1,   -1,  std::sqrt(2)),
          Node(1,    1,  std::sqrt(2))};
}

void a_star_planning(float sx, float sy,
                     float gx, float gy,
                     vector<float> ox_, vector<float> oy_,
                     float reso, float rr)
{
	//创建起点和终点
  Node* nstart = new Node((int)std::round(sx/reso), (int)std::round(sy/reso), 0.0); //reso表示分辨率，即最小的方格大小; sx，sy表示出发点的横纵坐标
  Node* ngoal = new Node((int)std::round(gx/reso), (int)std::round(gy/reso), 0.0);

  // 将障碍物坐标转换为整数，并计算最小和最大的 x、y 坐标值
  vector<int> ox;
  vector<int> oy;
  
  int min_ox = std::numeric_limits<int>::max();
  int max_ox = std::numeric_limits<int>::min();
  int min_oy = std::numeric_limits<int>::max();
  int max_oy = std::numeric_limits<int>::min();


  for(float iox:ox_){
      int map_x = (int)std::round(iox*1.0/reso);
      ox.push_back(map_x);
      min_ox = std::min(map_x, min_ox);
      max_ox = std::max(map_x, max_ox);
  }

  for(float ioy:oy_){
      int map_y = (int)std::round(ioy*1.0/reso);
      oy.push_back(map_y);
      min_oy = std::min(map_y, min_oy);
      max_oy = std::max(map_y, max_oy);
  }
  // 计算地图的宽度和高度
  int xwidth = max_ox-min_ox;
  int ywidth = max_oy-min_oy;
  //cout << "xwidth in A_star_main" << xwidth << endl;
  //visualization  创建用于可视化的背景图像
  cv::namedWindow("astar", cv::WINDOW_NORMAL);
  int count = 0;
  int img_reso = 4;
  cv::Mat bg(img_reso*ywidth,
             img_reso*ywidth,
             CV_8UC3,
             cv::Scalar(255,255,255));
    // 在背景图上绘制起点和终点的矩形
    cv::rectangle(bg,
                  cv::Point(nstart->x*img_reso+1, nstart->y*img_reso+1),
                  cv::Point((nstart->x+1)*img_reso, (nstart->y+1)*img_reso),
                  cv::Scalar(255, 0, 0), -1);
    cv::rectangle(bg,
                  cv::Point(ngoal->x*img_reso+1, ngoal->y*img_reso+1),
                  cv::Point((ngoal->x+1)*img_reso, (ngoal->y+1)*img_reso),
                  cv::Scalar(0, 0, 255), -1);
  // 创建访问地图和路径代价地图
  std::vector<std::vector<int> > visit_map(xwidth, vector<int>(ywidth, 0));
  
  std::vector<std::vector<float> > path_cost(xwidth, vector<float>(ywidth, std::numeric_limits<float>::max()));

  path_cost[nstart->x][nstart->y] = 0;
  // 计算障碍物地图，并在背景图上绘制障碍物
  std::vector<std::vector<int> > obmap = calc_obstacle_map(
                                                  ox, oy,
                                                  min_ox, max_ox,
                                                  min_oy, max_oy,
                                                  reso, rr,
                                                  bg, img_reso);

  auto cmp = [](const Node* left, const Node* right){return left->sum_cost > right->sum_cost;};
  std::priority_queue<Node*, std::vector<Node*>, decltype(cmp)> pq(cmp);
  // 将起点加入优先队列
  pq.push(nstart);
  std::vector<Node> motion = get_motion_model();
  // A*算法的主循环
  while (true)
  {
    if(SINGLE_STEP_MODE)
    {
      char c = getchar();
      if(c == EOF) exit(0);
    }
    else {};
    Node * node = pq.top();
    // 如果节点已访问过，则忽略
    if (visit_map[node->x-min_ox][node->y-min_oy] == 1){                  
      pq.pop();
      delete node;
      continue;
    }
	  else
    {
      pq.pop();
      visit_map[node->x-min_ox][node->y-min_oy] = 1;
    }
	// 如果当前节点是目标节点，则记录最终代价并结束循环
    if (node->x == ngoal->x && node->y==ngoal->y){
      ngoal->sum_cost = node->sum_cost;
      ngoal->p_node = node;
      break;
    }
	// 遍历所有可能的运动模式
    for(int i=0; i<motion.size(); i++){
      Node * new_node = new Node(
        node->x + motion[i].x,
        node->y + motion[i].y,
        path_cost[node->x][node->y] + motion[i].sum_cost + calc_heuristic(ngoal, node),
        node);
	  // 检查新节点是否合法
      if (!verify_node(new_node, obmap, min_ox, max_ox, min_oy, max_oy)){
        delete new_node;
        continue;
      }
	  // 检查新节点是否已访问过
      if (visit_map[new_node->x-min_ox][new_node->y-min_oy]){
        delete new_node;
        continue;
      }
      else{

      }
	  // 在背景图上绘制新节点
      cv::rectangle(bg,
                    cv::Point(new_node->x*img_reso+1, new_node->y*img_reso+1),
                    cv::Point((new_node->x+1)*img_reso, (new_node->y+1)*img_reso),
                    cv::Scalar(0, 255, 0));
      count++;
      cv::imshow("astar", bg);
      cv::waitKey(1);
	  // 更新路径代价并将新节点加入优先队列
      if (path_cost[node->x][node->y] + motion[i].sum_cost < path_cost[new_node->x][new_node->y]){
        path_cost[new_node->x][new_node->y] = path_cost[node->x][node->y] + motion[i].sum_cost; 
        pq.push(new_node);
      }
    }
      cv::rectangle(bg,
                    cv::Point(node->x*img_reso+1, node->y*img_reso+1),
                    cv::Point((node->x+1)*img_reso, (node->y+1)*img_reso),
                    cv::Scalar(0, 0, 255));    
                    //cout << "CLOSE :  " << node -> x << " " <<  node -> y << endl;
  }
  // 计算最终路径并绘制到背景图上
  calc_final_path(ngoal, reso, bg, img_reso);
  delete ngoal;
  delete nstart;
  cv::imshow("astar", bg);
  cv::waitKey(0);
};

int main(){
  // float sx = 10.0;
  // float sy = 10.0;
  // float gx = 80.0;
  // float gy = 170.0;
  float sx = 10.0;
  float sy = 10.0;
  float gx = 140.0;
  float gy = 210.0;
    cv::Mat img = cv::imread("..//..//pic//bj.jpg");
    // cv::Mat img = cv::imread("/..//pic//map.jpg");

// 筛选色域范围
    cv::Scalar lower_blue = cv::Scalar(80, 160, 130);  
    cv::Scalar upper_blue = cv::Scalar(120, 210, 150);  
    // cv::Scalar lower_blue = cv::Scalar(220, 200, 130);  
    // cv::Scalar upper_blue = cv::Scalar(255, 230, 180);  
    // imshow("src_img", img);
    // cv::waitKey(0);
    // 蓝色掩罩
    cv::Mat blue_mask;
    cv::inRange(img, lower_blue, upper_blue, blue_mask);
    cv::imshow("mask", blue_mask);
    cv::waitKey(0);

    //膨胀
    cv::Mat kernel_dilate = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(4, 4));
    cv::Mat dilated_mask;
    cv::dilate(blue_mask, dilated_mask, kernel_dilate);
    cv::imshow("dilated_mask",dilated_mask);
    cv::waitKey(0);
  if (img.empty()) {
      std::cout << "Failed to read the image." << std::endl;
  }
    // 将图像转换为栅格数据结构
  // GridData grid = convertImageToGrid(blue_mask);  
  GridData grid = convertImageToGrid(dilated_mask);  
    
  float grid_size = 1.0;
  float robot_size = 1.0;

  vector<float> ox;
  vector<float> oy;

  // 假设grid为二维数组表示的栅格地图
  float reso = 1.0, img_reso = 1.0;
  std::vector<std::vector<int>> map = create_map_from_grid(grid, reso, img, img_reso);
  
  
  
  while(grid[sx][sy])
  {
    cout << "Illegal start point!" << "input again with space :" << endl;
    cin >> sx >> sy;
  }
  while(grid[gx][gy])
  {
    cout << "Illegal goal point!" << "input again with space :" << endl;
    cin >> gx >> gy;
  }

  int map_x = map.size();
  int map_y = map[0].size();
  cout << map_x << endl;
    cout << map_y << endl;
  for(float i=0; i<map_y; i++){
    ox.push_back(i);
    oy.push_back(1.0*map_y);
  }
  for(float i=0; i<map_x; i++){
    ox.push_back(1.0*map_x);
    oy.push_back(i);
  }
  for(float i=0; i<map_y + 1; i++){
    ox.push_back(i);
    oy.push_back(map_y*1.0);
  }
  for(float i=0; i<map_x+1; i++){
    ox.push_back(0.0);
    oy.push_back(i);
  }
  cout << "grid.size() : " << grid.size() << endl;
  cout << "grid[0].size() : " << grid[0].size() << endl;
  for (int i = 0; i < grid.size(); ++i) {
        for (int j = 0; j < grid[i].size(); ++j) {
            if(grid[i][j] == 1)
            {
              ox.push_back((float)i);
              oy.push_back((float)j);
            }
        }
    }
  a_star_planning(sx, sy, gx, gy, ox, oy, grid_size, robot_size);
  return 0;
}