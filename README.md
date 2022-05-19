# :smile:Numpy暴力手撕深度神经网络
  一个**沉迷于炼丹术**的**人工智能**专业学生:smirk:  
  如果您对我的代码感兴趣，欢迎**fork或issue**！
# :sparkles:环境要求
`numpy==1.19.2`  
`matplotlib==3.1.3`
# 第一步，先clone我的代码
**方法一**：直接download_zip，然后解压  
**方法二**：cmd里输入`git clone https://github.com/AIHHU/Numpy-DNN`
# 第二步，配环境
## :grimacing:我觉得这俩大多数人都有，如果没有，直接cmd里打
`cd Numpy-DNN-main`  
`pip install -r requirements.txt`  
**稍微解释一下**：首先进入项目文件夹，然后pip需要安装的软件包  
**关于requirements.txt怎么来的？**：网上查一查pipreqs就知道了
# 第三步，直接运行
在cmd环境下：  
:sunny:如果您对用**单隐层神经网络处理线性回归**感兴趣，输入以下代码：  
`python LR_demo.py`  
:sunny:如果您对用**深度神经网络处理线性回归**感兴趣，输入以下代码：  
`python DNN_demo.py`  
:smiling_imp:**注意**:使用**深度神经网络处理线性回归**的意思是，使用了**非线性结构**去拟合**线性**的数据集，因为这里我只写了一个线性数据的生成脚本，所以这里只做测试用，证明所写深度神经网络是**有效**的
# 第四步，查看结果
可以在命令行中看到训练的**loss日志**，训练结束后，可以看到**loss曲线的window**  

:smiley:最终Loss曲线我也上传在了sample文件夹内，感兴趣可以看看
# 如果您单纯地对线性回归感兴趣
可以访问我的AI studio项目：[numpy手撕线性回归](https://www.baidu.com/)
