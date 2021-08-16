.. _dataset-guide:

=======================
使用 Dataset 定义数据集
=======================

.. note::

   * :py:class:`~.DataLoader` 初始化时必须提供 ``dataset`` 参数，通过传入一个数据集对象，表明将要加载的数据；
   * 这个世界上的数据集五花八门，并且总是以各种格式分布在不同地方，或许一些数据集形式已经成为了参考标准，
     但并不是所有数据一开始都用 MegEngine 所支持格式的进行存储，
     很多时候需要我们写脚本借助一些库对原始数据进行处理，并且转换成 MegEngine 中可用的 ``Dataset``.
   * MegEngine 中可以 :ref:`megengine-dataset` （如 :py:class:`~.PascalVOC`,  :py:class:`~.ImageNet` 等）
     替用户完成数据集的获取、切分等工作。但一些时候我们需要使用自己采集和标注好的的数据集，
     因此在使用 ``DataLoader`` 之前，通常需要将要用到的数据集人为地封装。

.. _dataset-type:

数据集类型
----------

根据样本的访问方式， MegEngine 中的数据集类型可分为 Map-style 和 Iterable-style 两种：

.. list-table:: 

   * - 类型 
     - :ref:`map-style-dataset`
     - :ref:`iterable-style-dataset`
   * - 抽象基类 [1]_
     - :py:class:`~.dataset.Dataset` / :py:class:`~.dataset.ArrayDataset`
     - :py:class:`~.dataset.StreamDataset`
   * - 访问方式
     - 支持随机
     - 顺序迭代
   * - 适用情景
     - Numpy 数组、字典、磁盘文件 [2]_
     - 生成器、来自网络的流数据

.. [1] 不可直接实例化使用，对应类型的数据集必须作为子类继承，并实现必需的协议。
.. [2] 通常应该尽可能使用 Map-style 的数据集。它提供了查询数据集的大小的方法，更容易打乱，并能够轻松地并行加载。
       但 ``MapDataset`` 不太适合输入数据作为流的一部分到达的情况，例如音频或视频源；
       或者每个数据点可能是文件的一个子集，该文件太大而无法保存在内存中，因此需要在训练期间进行增量加载。
       虽然这些情况可以通过我们数据集中更复杂的逻辑或我们输入的额外预处理来解决，
       但现在有一个更自然的解决方案，即使用 ``IterableDataset`` 作为输入。

.. _map-style-dataset:

Map-style
~~~~~~~~~

:py:class:`~.dataset.Dataset` （也可被称为 ``MapDataset`` ）
  MegEngine 中所有数据集的抽象基类。
  对应数据集类型为 Map-style, 即表示从索引/键到数据样例的映射，具有随机访问能力。
  例如使用 ``dataset[idx]`` 可以从磁盘上的文件夹中读取到第 ``idx`` 个图像及其相应的标签。
  使用时需要实现 ``__getitem__()`` 和 ``__len__()`` 协议。

  下面的代码展示了如何生成一个从 0 到 5 的数据集（不带标签）：

  .. code-block:: python

     from megengine.data.dataset import Dataset

     class CustomMapDataset(Dataset):
         def __init__(self, data):
             self.data = data

         def __getitem__(self, idx):
             return self.data[idx]

         def __len__(self):
             return len(self.data)

  >>> data = list(range(0, 5))
  >>> dataset = CustomMapDataset(data)
  >>> print(len(dataset))
  >>> print(dataset[2])
  5
  2

  .. warning::

     请注意，为了避免在加载大型数据集时一次性将数据加载到内存导致 OOM（Out Of Memory），
     我们建议将实际的数据读取操作实现在 ``__getitem__`` 方法中，而不是 ``__init__`` 方法中，
     后者仅记录映射关系中的索引/键内容（可能是文件名或路径组成的列表），这可以极大程度地减少内存占用。
     具体的例子可参考 :ref:`load-image-data-example` 。

:py:class:`~.dataset.ArrayDataset`
  对 ``Dataset`` 类的进一步封装，适用于 NumPy ndarray 数据，无需实现 ``__getitem__()`` 和 ``__len__()`` 协议。

  下面的代码展示了如何生成随机一个具有 100 个样本，每张样本为 32 x 32 像素的 RGB 图片的数据集（标签为随机值）
  这也是我们在处理图像时经常遇到的 ``(N, C, H, W)`` 格式：

  .. code-block:: python

     import numpy as np
     from megengine.data.dataset import ArrayDataset

     data = np.random.random((100, 3, 32, 32))
     target = np.random.random((100, 1))
     dataset = ArrayDataset(data, target)

  >>> print(len(dataset))
  >>> print(type(dataset[0]), len(dataset[0])) 
  >>> print(dataset[0][0].shape)
  100
  <class 'tuple'> 2
  (3, 32, 32)

.. _iterable-style-dataset:
	
Iterable-style
~~~~~~~~~~~~~~

:py:class:`~.dataset.StreamDataset` （也可被称为 ``IterableDataset`` ）
  Iterable-style 数据集，适用于流式数据，即迭代式地访问数据，
  例如使用 ``iter(dataset)`` 可以返回从数据库、远程服务器甚至实时生成的日志中读取的数据流，
  然后不断使用 ``next`` 迭代从而实现遍历。这种类型的数据集特别适用于随机读取成本过高甚至不可能的情况，
  以及批量大小取决于获取的数据的情况。使用时需要实现 ``__iter__()`` 协议。

  下面的代码展示了如何生成一个从 0 到 5 的数据集（不带标签）：

  .. code-block:: python

     from megengine.data.dataset import StreamDataset

     class CustomIterableDataset(StreamDataset):
         def __init__(self, data):
             self.data = data

         def __iter__(self):
             return iter(self.data)

  >>> data = list(range(0, 5))
  >>> dataset = CustomIterableDataset(data)
  >>> curr = next(iter(dataset))
  >>> print(curr)
  >>> for data in iter(dataset):
  ...     print(data)
  0
  0
  1
  2
  3
  4

  .. warning::

     尝试计算 ``StreamDataset`` 的长度（调用 ``__len__`` ）或随机访问其中的元素（调用 ``__getitem__`` ）
     都将会抛出相关报错信息。

.. _megengine-dataset:

使用已经实现的数据集接口
------------------------

在 :py:mod:`~.data.dataset` 子模块中，除了提供了一些抽象基类待用户自定义子类进行实现，
还提供了一些基于主流数据集封装好的接口，比如常被用于教学和练习用途的 :py:class:`~.MNIST` 数据集：

>>> from megengine.data.dataset import MNIST
>>> train_set = MNIST(root="path/to/data/", train=True, download=False)
>>> test_set = MNIST(root="path/to/data/", train=False, download=False)

借助上面的代码，我们可以快速的获取 MNIST 数据集的训练集 ``train_set`` 和测试集 ``test_set`` ，
其中 ``download`` 参数可以控制是否要从数据集官方提供的地址进行下载。更多细节请参考 API 文档。

.. note::

   * 一些数据集由于许可协议中的规定将不提供原始数据的下载接口（如 :py:class:`~.ImageNet` ），需手动下载；
   * 用户也可以选择使用其它的库、脚本或者工具下载原始数据集，
     注意下载速度将受到本地和访问服务器带宽和网络环境的影响；
     接着可以将下载好的数据集放置于对应路径，再使用 MegEngine 提供的 API 进行后续处理。

.. _how-to-add-datasets:

如何添加新的数据集
------------------

该部分的内容尚在建设中...
