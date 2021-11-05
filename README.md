# 基于DMS的Q-Learning
## 介绍
- Q-Learning：强化学习的一种，基于一个Q表来学习在不同的State下采取不同的Action取得的未来期望奖励，从而学习到Action的策略。
- 基于DMS的Q-Learning：结合了DMS(Dynamic Model Selection)的Q-Learning，首先构建一个预测模型池，Agent可以学习到从模型池里选择模型的策略，
可以提高预测准确率。
## 技术用途
- 可用于基于时序的数据预测。
## 方法实现
- ANN_train/LSTM_train/...：以训练集预训练模型ANN/LSTM/MSVR/HoltWinters，并以.h5py文件或.txt文件为形式保存其参数，用于测试集数据的预测。
- ANNhpSelecting/LSTMhpSelecting/...：基于Grid Search使用不同的超参数组合来训练并验证各个模型，通过输出.xlsx表格文件分析最优参数组合是哪一组。
- DMS_test_projectA：DMS测试的主程序，使用由4个模型，每个模型各使用两组最优超参数来组成共有8个模型的模型池。Agent的训练集T由滑动窗口取得，
然后基于Agent的训练集T选出最优的I个候选模型，Agent采用\epsilon-greedy的方式为下个time step从候选模型中选取模型用于预测下个time step的数值，
State为当前time step下Agent所用模型，Action为Agent为下一个time step选择的模型，计算Reward并更新Q表。最终Agent结束训练阶段后，依据Q表，
即学习到的经验，应用于DMS处理数据集D。
## 实验结果展示
首先输出模型池中8种模型的预测值、DMS based Q-Learning预测值和真实值的比较，如下图所示。可以从直观上来说，DMS based Q-Learning预测值在整条序列上更靠近真实值。  
![DMS](https://github.com/FFFjx/DMS-QL/blob/main/pic/DMS-result1.png)  
然后输出程序结果：  
![DMS](https://github.com/FFFjx/DMS-QL/blob/main/pic/DMS-result2.png)  
best action sequence表示测试集中每个time step理论上最优Action所代表的模型序号；DMS action sequence表示算法在测试集中累积选择的模型序号；
best rank sequence表示测试集中每个time step理论上最优Action所代表的模型排名，数值越小表示越优，1为最优；DMS rank sequence表示算法在测试集
中累积选择的模型对应的排名；best mape sequence表示测试集中每个time step理论上最优Action所代表的模型在该time step的mape；DMS mape sequence
算法在测试集中累积选择的模型在该time step的mape；剩下的分别是模型池中8种模型和DMS算法在整段测试集中的mape和arv值，mape和arv都是预测评价指标，
数值越小表示预测效果越好，best choice则表示理论上能取到的上限。可以从结果中得出，虽然Agent还不能学习到全局最优的策略，但是对比起任意一个单一模型
来说都有优势，相当于提高了单一模型的预测准确率。