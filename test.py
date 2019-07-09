'''

@File    :   test.py 
@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2019-07-05 13:18   shenhangke      1.0         None
---------------------


 np.array.__array_interface__  返回一个字典，其中包含具体的数组指针？
 这个字典中的成员参见：https://www.numpy.org.cn/reference/array_objects/array_interface.html


'''

import transE
import ctypes
import tensorflow as tf
import numpy as np

"""
config = transE.Config()
ll = ctypes.cdll.LoadLibrary
lib = ll("./libMac/libTransElib.dylib")

lib.init()
config.relation = lib.getRelationTotal()
config.entity = lib.getEntityTotal()
config.batch_size = lib.getTripleTotal() // config.nbatches

# the count of data
print(config.relation)  # 1345
print(config.entity)  # 14951
print(config.batch_size)  # 4831


entity_total = config.entity  # 实体数目？
relation_total = config.relation
batch_size = config.batch_size
size = config.hidden_size
margin = config.margin  # 边界值

pos_h = tf.placeholder(tf.int32, [None])  # 这个应该是正确的三元组中头节点的输入
pos_t = tf.placeholder(tf.int32, [None])  # 以下对应关系和尾实体
pos_r = tf.placeholder(tf.int32, [None])

neg_h = tf.placeholder(tf.int32, [None])  # 错误实体的头节点输入，以下通用
neg_t = tf.placeholder(tf.int32, [None])
neg_r = tf.placeholder(tf.int32, [None])

with tf.name_scope("embedding"):
    ent_embeddings = tf.get_variable(name="ent_embedding", shape=[entity_total, size],
                                          initializer=tf.contrib.layers.xavier_initializer(uniform=False))
    # 同理，这里是关系相关的变量
    rel_embeddings = tf.get_variable(name="rel_embedding", shape=[relation_total, size],
                                          initializer=tf.contrib.layers.xavier_initializer(uniform=False))
    pos_h_e = tf.nn.embedding_lookup(ent_embeddings, pos_h)
    pos_t_e = tf.nn.embedding_lookup(ent_embeddings, pos_t)
    pos_r_e = tf.nn.embedding_lookup(rel_embeddings, pos_r)
    neg_h_e = tf.nn.embedding_lookup(ent_embeddings, neg_h)
    neg_t_e = tf.nn.embedding_lookup(ent_embeddings, neg_t)
    neg_r_e = tf.nn.embedding_lookup(rel_embeddings, neg_r)

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    print(sess.run(ent_embeddings))
    print(sess.run(pos_h_e))
    #print(ent_embeddings)
    """

# 首先现在我想知道的是，从底层数据文件中获取到的数据是什么数据
# 类字段在公共区域保存一份，对象字段在每个对象中都保存一份
# 所以在python里面还可以直接用对象+.来直接创建一个对像属性

# 获取由底层c++提供的数据

ll = ctypes.cdll.LoadLibrary
lib = ll("./libMac/libTransElib.dylib")
lib.setInPath("./data/FB15K/".encode("ascii"))
lib.setBernFlag(0)
lib.init()

batch = 100
ph = np.zeros(batch, dtype=np.int32)
pt = np.zeros(batch, dtype=np.int32)
pr = np.zeros(batch, dtype=np.int32)
nh = np.zeros(batch, dtype=np.int32)
nt = np.zeros(batch, dtype=np.int32)
nr = np.zeros(batch, dtype=np.int32)

ph_addr = ph.__array_interface__['data'][0]  # use this interface return a pointer which point to array
pt_addr = pt.__array_interface__['data'][0]
pr_addr = pr.__array_interface__['data'][0]
nh_addr = nh.__array_interface__['data'][0]
nt_addr = nt.__array_interface__['data'][0]
nr_addr = nr.__array_interface__['data'][0]

lib.getBatch.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                     ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
lib.getBatch(ph_addr, pt_addr, pr_addr, nh_addr, nt_addr, nr_addr, batch)

print(ph)
