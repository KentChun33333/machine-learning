# 载入此项目所需要的库
import numpy as np
import pandas as pd
import visuals as vs # Supplementary code
from sklearn.cross_validation import ShuffleSplit

import matplotlib.pyplot as plt

# Pretty display for notebooks
# 让结果在notebook中显示
%matplotlib inline

# Load the Boston housing dataset
# 载入波士顿房屋的数据集
data = pd.read_csv('housing.csv')
prices = data['MEDV']
features = data.drop('MEDV', axis = 1)
    
# Success
# 完成
print "Boston housing dataset has {} data points with {} variables each.".format(*data.shape)

# TODO: Minimum price of the data
#目标：计算价值的最小值
minimum_price = None
minimum_price = min(prices)

# if want to return index : np.argmin(prices)

# TODO: Maximum price of the data
#目标：计算价值的最大值
maximum_price = None
maximum_price = max(prices)
# TODO: Mean price of the data
#目标：计算价值的平均值
mean_price = None
mean_price = np.average(prices)

# TODO: Median price of the data
#目标：计算价值的中值
median_price = None
median_price = np.median(prices)

# TODO: Standard deviation of prices of the data
#目标：计算价值的标准差
std_price = None
std_price = np.std(prices)

# Show the calculated statistics
#目标：输出计算的结果
print "Statistics for Boston housing dataset:\n"
print "Minimum price: ${:,.2f}".format(minimum_price)
print "Maximum price: ${:,.2f}".format(maximum_price)
print "Mean price: ${:,.2f}".format(mean_price)
print "Median price ${:,.2f}".format(median_price)
print "Standard deviation of prices: ${:,.2f}".format(std_price)



from sklearn import metrics

# metrics.mean_squared_error example
# SSE = sum of squared_error
# mean_squared_error = SSE/(sum of samples-numbers)
# High bias = less feature = underfit 
# High varian = more feature = overfit
# good model = large R^2 = low SSE => as generated model as possible might be shinking the features input in some case.

# Data Type 3
## Numeric data (cound be Discrete or continues ) not ordered by time 
## Category data ( can have numberic, but no math-meaning for avg or ... etc [EX] 2012,2013, 1stars or  )
## Categorical data represent characteristic
## Categorical data could be ordered such as 1 stars to 5 stars of some evaluations
## Time Series Data is numeric data orderd by time


from sklearn import cross_validation
### iris = datasets.load_iris()
### features = iris.data
### labels = iris.target
### TrX, TeX, TrY, TeY = cross_validation.train_test_split(features, labels, test_size=0.4, random_state=0)
### TrX, TeX, TrY, TeY = cross_validation.train_test_split(X, Y, test_size=0.4, random_state=0)

### K-fold Cross-Validation 
from sklearn.cross_validation import KFold
kf = KFold(total_rows_of_TrX, k_vaule)
# Run K seperate .. experiment 
# preety much use all of the data to train our model, more accuracy, the only cons is spending more time to training our model.
# Tips of using KFOLD, be aware of each K-container would have highly bias behaviour
# we need to do some randomized for each K-container 
# so k is not as large as good ~also think about sampling-distribution issue
# 



################
# GridSearchCV #
################
sklearn.grid_search.GridSearchCV
# GridSearchCV 用于系统地遍历多种参数组合，通过交叉验证确定最佳效果参数。它的好处是，只需增加几行代码，就能遍历多种组合
# Help fast tuning hyperparameters of the model
# Usage Sample 
from sklearn import svm, grid_search, datasets
iris = datasets.load_iris()
parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}

#这时，会自动生成一个不同（kernel、C）参数值组成的“网格”:
#('rbf', 1)	('rbf', 10)
#('linear', 1)	('linear', 10)

svr = svm.SVC()
各组合均用于训练 SVM，并使用交叉验证对表现进行评估。

svr = svm.SVC() 
这与创建分类器有点类似，就如我们从第一节课一直在做的一样。
但是请注意，“clf” 到下一行才会生成—这儿仅仅是在说采用哪种算法。
另一种思考方法是，“分类器”在这种情况下不仅仅是一个算法，而是算法加参数值。
请注意，这里不需对 kernel 或 C 做各种尝试；下一行才处理这个问题。

clf = grid_search.GridSearchCV(svr, parameters) 
这是第一个不可思议之处，分类器创建好了。 
我们传达算法 (svr) 和参数 (parameters) 字典来尝试，它生成一个网格的参数组合进行尝试。

clf.fit(iris.data, iris.target) 
第二个不可思议之处。 拟合函数现在尝试了所有的参数组合，并返回一个合适的分类器，
自动调整至最佳参数组合。现在您便可通过 clf.best_params_ 来获得参数值。
clf = grid_search.GridSearchCV(svr, parameters)
clf.fit(iris.data, iris.target)                        
# GridSearchCV(cv=None, error_score=...,
#        estimator=SVC(C=1.0, cache_size=..., class_weight=..., coef0=...,
#                      decision_function_shape=None, degree=..., gamma=...,
#                      kernel='rbf', max_iter=-1, probability=False,
#                      random_state=None, shrinking=True, tol=...,
#                      verbose=False),
#        fit_params={}, iid=..., n_jobs=1,
#        param_grid=..., pre_dispatch=..., refit=...,
#        scoring=..., verbose=...)

#######################################################
# Faces recognition example using eigenfaces and SVMs #
#######################################################
# ref D:udacityBOSton
5 values of C 
6 values of gamma are test out 


###########################
# Curse of Dimensionality #
###########################

As the number of Features or Dimensionality grows, The amount of Data we need to generize accuracy grows enponential
KNN with line => plan => cubic 
to fill the data-space with more and more data 
so we need some 
# Bellman, Bellman function , Dynamical program inventor 

#######################################
# Learning curves of Model complexity #
#######################################

# Learning curve 
# 让我们根据模型通过可视化图形从数据中学习的能力来探讨偏差与方差之间的关系
# Bias and Variance 
机器学习中的学习曲线是一种可视化图形，能根据一系列训练实例中的训练和测试数据比较模型的指标性能。
在查看数据与误差之间的关系时，我们通常会看到，随着训练点数量的增加，误差会趋于下降。
由于我们尝试构建从经验中学习的模型，因此这很有意义。
我们将训练集和测试集分隔开，以便更好地了解能否将模型泛化到未见过的数据而不是拟合到刚见过的数据。

在学习曲线中，当训练曲线和测试曲线均达到稳定阶段，并且两者之间的差距不再变化时，则可以确认模型已尽其所能地了解数据。

偏差
在训练误差和测试误差收敛并且相当高时，这实质上表示模型具有偏差。
无论我们向其提供多少数据，模型都无法表示基本关系，因而出现系统性的高误差。

方差(overfitting)
如果训练误差与测试误差之间的差距很大，这实质上表示模型具有高方差。
与偏差模型不同的是，如果有更多可供学习的数据，或者能简化表示数据的最重要特征的模型，则通常可以改进具有方差的模型。

理想的学习曲线
I think it usually refers to a plot of the prediction accuracy/error vs. the training set size 
(ie: how better does the model get at predicting the target as you the increase number of instances used to train it)
模型的最终目标是，误差小并能很好地泛化到未见过的数据（测试数据）。
如果测试曲线和训练曲线均收敛，并且误差极低，就能看到这种模型。
In the right plot, we have the learning curve for d = 20. 
From the above discussion, we know that d = 20 is a high-variance estimator which over-fits the data.
This is indicated by the fact that the training error is much less than the cross-validation error. 
这种模型能根据未见过的数据非常准确地进行预测。

模型复杂度 (Ref http://www.astroml.org/sklearn_tutorial/practical.html#astro-biasvariance )
与学习曲线图形不同，模型复杂度图形呈现的是模型复杂度如何改变训练曲线和测试曲线，而不是用以训练模型的数据点的数量。
一般趋势是，随着模型增大，模型对固定的一组数据表现出更高的变化性。

学习曲线与模型复杂度
那么，学习曲线与模型复杂度之间有何关系？

如果我们获取具有同一组固定数据的相同机器学习算法的学习曲线，
但为越来越高的模型复杂度创建几个图形，则所有学习曲线图形均代表模型复杂度图形。
这就是说，如果我们获取了每个模型复杂度的最终测试误差和训练误差，
并依据模型复杂度将它们可视化，则我们能够看到随着模型的增大模型的表现有多好。

模型复杂度的实际使用
既然知道了能通过分析模型复杂度图形来识别偏差和方差的问题，现在可利用一个可视化工具来帮助找出优化模型的方法。
在下一部分中，我们会探讨 gridsearch 和如何微调模型以获得更好的性能。


项目概述
在此项目中，我们将对为马萨诸塞州波士顿地区的房屋价格收集的数据应用基本机器学习概念，以预测新房屋的销售价格。您首先将探索这些数据以获取数据集的重要特征和描述性统计信息。接下来，您要正确地将数据拆分为测试数据集和训练数据集，并确定适用于此问题的性能指标。然后，您将使用不同的参数和训练集大小分析学习算法的性能图表。这让您能够挑选最好地泛化到未见过的数据的最佳模型。最后，您将根据一个新样本测试此最佳模型并将预测的销售价格与您的统计数据进行比较。

使用模型评估和验证中的评估指标小节为此项目做准备。

项目亮点
此项目旨在让您熟练地在 Python 中处理数据集并使用 NumPy 和 Scikit-Learn 应用基本机器学习技术。在使用 sklearn 库中的许多可用算法之前，先练习分析和解释您模型的表现可能有所帮助。

通过完成此项目您将会学习（并最终知道）以下知识：

如何使用 NumPy 调查数据集的潜在特征。
如何分析各种学习性能图以了解方差和偏差。
如何确定用于从未看见的数据进行预测的最佳猜测模型。
如何评估模型使用之前的数据处理未看见数据的表现。

项目描述
波士顿房屋市场竞争异常激烈，您想成为当地最好的房地产中介。为了与同行竞争，您决定使用几个基本的机器学习概念来帮助您和客户找到其房屋的最佳销售价格。幸运的是，您遇到了波士顿房屋数据集，其中包含大波士顿社区的房屋的各种特征的累积数据，包括其中各个地区的房屋价格的中值。您的任务是利用可用工具基于统计分析来构建一个最佳模型。然后使用该模型估算客户房屋的最佳销售价格。

对于此任务，请在这里下载“boston_housing.ipynb”文件。虽然已经实现了某些代码以便于您开始操作，但您仍需要实现其他功能才能成功地回答 notebook 中包含的所有问题。可在以下幻灯片上找到包含的问题以供参考。除非我们要求，否则不要修改已包含的代码。

您还可以访问我们的项目 GitHub 以便访问此纳米学位的所有项目。

软件和库
对于此项目，您需要安装以下软件和 Python 库：

Python 2.7
NumPy
scikit-learn
iPython Notebook
可交付成果
提交的文件中应包含以下文件并且可以打包为单个 .zip 文件：

包含完整实现且可正常运行的代码的 “boston_housing.ipynb” 文件，并已执行所有代码块和显示了输出。
PDF 格式的项目报告（可以直接将 notebook 另存为包含答案的 PDF，或者提交只包含问题和答案的单独文档）。

项目报告问题
可在此处 找到波士顿房屋数据集的描述。使用此幻灯片作为您在 notebook 中遇到的项目问题的参考。随附的 PDF 报告中必须包含这些问题（以及您的答案）。

统计分析与数据研究
此问题将整合到项目 notebook 输出中。
使用 NumPy 库计算有关此数据集的几个有意义的统计数据：

收集了多少个数据点（房屋）？
每所房屋存在多少个特征？
最低房屋价格是多少？最高价格是多少？
房屋价格的均值是多少？中间值是多少？
所有房屋价格的标准偏差是多少？
1) 在给定房屋的可用特征中，选择您认为比较重要的三个并简要描述它们衡量的方面。

2) 使用模板代码中客户的特征集“CLIENT_FEATURES”，哪些值与所选特征对应？

评估模型性能
3) 为什么要将数据拆分成训练和测试子集？

4) 下面哪个性能指标最适合预测房屋价格和分析错误？为什么？

准确率
精确率
召回率
F1 分数
均方误差 (MSE)
平均绝对误差 (MAE)
5) 什么是网格搜索算法？它在哪些情况下适用？

6) 什么是交叉验证？对模型使用交叉验证的效果如何？为什么在使用网格搜索时，交叉验证会有帮助？

分析模型性能
7) 选择您的代码创建的学习曲线图之一。模型的最大深度是多少？当训练集大小增加时，训练误差会出现什么情况？描述测试误差出现的情况。

8) 查看最大深度分别为 1 和 10 的模型的学习曲线图。当模型使用完整训练集时，如果最大深度为 1，它是否会出现高偏差或高方差？最大深度为 10 时呢？

9) 根据模型复杂度图表，描述当最大深度增加时的训练和测试误差。根据您对图表的解读，哪个最大深度会促使模型最好地泛化数据集？为什么？

模型预测
为了回答以下问题，建议您多次运行 notebook 并使用中间值和均值作为结果。

10) 使用网格搜索时，模型的最佳最大深度是多少？此结果与您最初的直观印象相比如何？

11) 使用参数经过调整的模型，客户房屋的最佳销售价格是多少？此销售价格与您基于数据集计算的统计数据相比如何？

12) 用几句话讨论您是否会使用此模型预测客户在波士顿地区未来房屋的销售价格。
评估
优达学城的项目导师会依据此评估准则评审您的简历项目。提交前请务必仔细查看该标准。所有标准必须“符合规格”才能通过。

提交
如果你是第一次在优达学城提交项目，点击提交之后，要等1分钟左右才能打开提交页面。如果长时间打不开，可以刷新。如果依然无法打开项目提交页面，可以联系客服微信或者邮件至 support@youdaxue.com

我们正在努力修复这个问题。

后续步骤
在项目导师给出反馈后，您会立即收到电子邮件。在此期间，请查看下一个项目，并尽管开始实施该项目或学习它的辅助课程！
F1 = 2 * (精确率 * 召回率) / (精确率 + 召回率)

####
# Decisoin Tree
####
def mean_squared_error(y_true, y_pred,
                       sample_weight=None,
                       multioutput='uniform_average'):
    """Mean squared error regression loss
    Read more in the :ref:`User Guide <mean_squared_error>`.
    Parameters
    ----------
    y_true : array-like of shape = (n_samples) or (n_samples, n_outputs)
        Ground truth (correct) target values.
    y_pred : array-like of shape = (n_samples) or (n_samples, n_outputs)
        Estimated target values.
    sample_weight : array-like of shape = (n_samples), optional
        Sample weights.
    multioutput : string in ['raw_values', 'uniform_average']
        or array-like of shape (n_outputs)
        Defines aggregating of multiple output values.
        Array-like value defines weights used to average errors.
        'raw_values' :
            Returns a full set of errors in case of multioutput input.
        'uniform_average' :
            Errors of all outputs are averaged with uniform weight.
    Returns
    -------
    loss : float or ndarray of floats
        A non-negative floating point value (the best value is 0.0), or an
        array of floating point values, one for each individual target.
    Examples
    --------
    >>> from sklearn.metrics import mean_squared_error
    >>> y_true = [3, -0.5, 2, 7]
    >>> y_pred = [2.5, 0.0, 2, 8]
    >>> mean_squared_error(y_true, y_pred)
    0.375
    >>> y_true = [[0.5, 1],[-1, 1],[7, -6]]
    >>> y_pred = [[0, 2],[-1, 2],[8, -5]]
    >>> mean_squared_error(y_true, y_pred)  # doctest: +ELLIPSIS
    0.708...
    >>> mean_squared_error(y_true, y_pred, multioutput='raw_values')
    ... # doctest: +ELLIPSIS
    array([ 0.416...,  1.        ])
    >>> mean_squared_error(y_true, y_pred, multioutput=[0.3, 0.7])
    ... # doctest: +ELLIPSIS
    0.824...
    """
    y_type, y_true, y_pred, multioutput = _check_reg_targets(
        y_true, y_pred, multioutput)
    output_errors = np.average((y_true - y_pred) ** 2, axis=0,
                               weights=sample_weight)
    if isinstance(multioutput, string_types):
        if multioutput == 'raw_values':
            return output_errors
        elif multioutput == 'uniform_average':
            # pass None as weights to np.average: uniform mean
            multioutput = None

    return np.average(output_errors, weights=multioutput)

# sklearn.metrics.r2_score R^2 (coefficient of determination) regression score function.
r2_score(y_true, y_pred)  








import numpy as np
import pylab as pl
from matplotlib import ticker
from matplotlib.patches import FancyArrow

# Suppress warnings from Polyfit (ill-conditioned fit)
import warnings
warnings.filterwarnings('ignore', message='Polyfit*')

np.random.seed(42)

def test_func(x, err=0.5):
    return np.random.normal(10 - 1. / (x + 0.1), err)


def compute_error(x, y, p):
    yfit = np.polyval(p, x)
    return np.sqrt(np.mean((y - yfit) ** 2))


#------------------------------------------------------------
# Plot linear regression example
np.random.seed(42)
x = np.random.random(20)
y = np.sin(2 * x)
p = np.polyfit(x, y, 1)  # fit a 1st-degree polynomial to the data

xfit = np.linspace(-0.2, 1.2, 10)
yfit = np.polyval(p, xfit)

pl.scatter(x, y, c='k')
pl.plot(xfit, yfit)
pl.xlabel('x')
pl.ylabel('y')
pl.title('Linear Regression Example')

#------------------------------------------------------------
# Plot example of over-fitting and under-fitting

N = 8
np.random.seed(42)
x = 10 ** np.linspace(-2, 0, N)
y = test_func(x)

xfit = np.linspace(-0.2, 1.2, 1000)

titles = ['d = 1 (under-fit)', 'd = 2', 'd = 6 (over-fit)']
degrees = [1, 2, 6]

pl.figure(figsize = (9, 3.5))
for i, d in enumerate(degrees):
    pl.subplot(131 + i, xticks=[], yticks=[])
    pl.scatter(x, y, marker='x', c='k', s=50)

    p = np.polyfit(x, y, d)
    yfit = np.polyval(p, xfit)
    pl.plot(xfit, yfit, '-b')
    
    pl.xlim(-0.2, 1.2)
    pl.ylim(0, 12)
    pl.xlabel('house size')
    if i == 0:
        pl.ylabel('price')

    pl.title(titles[i])

pl.subplots_adjust(left = 0.06, right=0.98,
                   bottom=0.15, top=0.85,
                   wspace=0.05)

#------------------------------------------------------------
# Plot training error and cross-val error
#   as a function of polynomial degree

Ntrain = 100
Ncrossval = 100
error = 1.0

np.random.seed(0)
x = np.random.random(Ntrain + Ncrossval)
y = test_func(x, error)

xtrain = x[:Ntrain]
ytrain = y[:Ntrain]

xcrossval = x[Ntrain:]
ycrossval = y[Ntrain:]

degrees = np.arange(1, 21)
train_err = np.zeros(len(degrees))
crossval_err = np.zeros(len(degrees))

for i, d in enumerate(degrees):
    p = np.polyfit(xtrain, ytrain, d)

    train_err[i] = compute_error(xtrain, ytrain, p)
    crossval_err[i] = compute_error(xcrossval, ycrossval, p)

pl.figure()
pl.title('Error for 100 Training Points')
pl.plot(degrees, crossval_err, lw=2, label = 'cross-validation error')
pl.plot(degrees, train_err, lw=2, label = 'training error')
pl.plot([0, 20], [error, error], '--k', label='intrinsic error')
pl.legend()
pl.xlabel('degree of fit')
pl.ylabel('rms error')

pl.gca().add_patch(FancyArrow(5, 1.35, -3, 0, width = 0.01,
                              head_width=0.04, head_length=1.0,
                              length_includes_head=True))
pl.text(5.3, 1.35, "High Bias", fontsize=18, va='center')

pl.gca().add_patch(FancyArrow(19, 1.22, 0, -0.1, width = 0.25,
                              head_width=1.0, head_length=0.05,
                              length_includes_head=True))
pl.text(19.8, 1.23, "High Variance", ha='right', fontsize=18)

#------------------------------------------------------------
# Plot training error and cross-val error
#   as a function of training set size

Ntrain = 100
Ncrossval = 100
error = 1.0

np.random.seed(0)
x = np.random.random(Ntrain + Ncrossval)
y = test_func(x, error)

xtrain = x[:Ntrain]
ytrain = y[:Ntrain]

xcrossval = x[Ntrain:]
ycrossval = y[Ntrain:]

sizes = np.linspace(2, Ntrain, 50).astype(int)
train_err = np.zeros(sizes.shape)
crossval_err = np.zeros(sizes.shape)

pl.figure(figsize=(10, 5))

for j,d in enumerate((3,40)):
    for i, size in enumerate(sizes):
        p = np.polyfit(xtrain[:size], ytrain[:size], d)
        crossval_err[i] = compute_error(xcrossval, ycrossval, p)
        train_err[i] = compute_error(xtrain[:size], ytrain[:size], p)

    ax = pl.subplot(121 + j)
    pl.plot(sizes, crossval_err, lw=2, label='cross-val error')
    pl.plot(sizes, train_err, lw=2, label='training error')
    pl.plot([0, Ntrain], [error, error], '--k', label='intrinsic error')

    pl.xlabel('traning set size')
    if j == 0:
        pl.ylabel('rms error')
    else:
        ax.yaxis.set_major_formatter(ticker.NullFormatter())
    
    pl.legend(loc = 4)
    
    pl.ylim(0.0, 2.5)
    pl.xlim(0, 99)

    pl.text(98, 2.45, 'd = %i' % d, ha='right', va='top', fontsize='large')

pl.subplots_adjust(wspace = 0.02, left=0.07, right=0.95,
                   bottom=0.1, top=0.9)
pl.show()