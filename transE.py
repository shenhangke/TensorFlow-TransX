# coding:utf-8
import numpy as np
import tensorflow as tf
import os
import time
import datetime
import ctypes

# 由于在底层的c++库中，所有的字符串都是用的ansii的编码方式，在python2.7中也是默认这个方式，但是到了python3，默认变成了utf
# 因此，所有有关于底层c++代码库的调用，都需要先转换编码，将编码转化为ansii再做下一步的处理。
# 需要双平台调试，所以windows平台和mac平台都需要一个动态链接库，Windows平台使用dll作为动态链接库

ll = ctypes.cdll.LoadLibrary
lib = ll("./libMac/libTransElib.dylib")
test_lib = ll("./libMac/libTransElibTest.dylib")


class Config(object):

    def __init__(self):
        lib.setInPath("./data/FB15K/".encode("ascii"))
        test_lib.setInPath("./data/FB15K/".encode("ascii"))
        lib.setBernFlag(0)
        self.learning_rate = 0.001
        self.testFlag = False
        self.loadFromData = False
        self.L1_flag = True
        self.hidden_size = 100
        self.nbatches = 100
        self.entity = 0 # 实体数目
        self.relation = 0 #关系数目
        self.trainTimes = 1000  #训练次数
        self.margin = 1.0  #边界值？


class TransEModel(object):

    def __init__(self, config):

        entity_total = config.entity  # 实体数目？
        relation_total = config.relation
        batch_size = config.batch_size
        size = config.hidden_size
        margin = config.margin  #边界值

        #一堆占位符

        self.pos_h = tf.placeholder(tf.int32, [None])
        self.pos_t = tf.placeholder(tf.int32, [None])
        self.pos_r = tf.placeholder(tf.int32, [None])

        self.neg_h = tf.placeholder(tf.int32, [None])
        self.neg_t = tf.placeholder(tf.int32, [None])
        self.neg_r = tf.placeholder(tf.int32, [None])


        with tf.name_scope("embedding"):
            self.ent_embeddings = tf.get_variable(name="ent_embedding", shape=[entity_total, size],
                                                  initializer=tf.contrib.layers.xavier_initializer(uniform=False))
            self.rel_embeddings = tf.get_variable(name="rel_embedding", shape=[relation_total, size],
                                                  initializer=tf.contrib.layers.xavier_initializer(uniform=False))
            pos_h_e = tf.nn.embedding_lookup(self.ent_embeddings, self.pos_h)
            pos_t_e = tf.nn.embedding_lookup(self.ent_embeddings, self.pos_t)
            pos_r_e = tf.nn.embedding_lookup(self.rel_embeddings, self.pos_r)
            neg_h_e = tf.nn.embedding_lookup(self.ent_embeddings, self.neg_h)
            neg_t_e = tf.nn.embedding_lookup(self.ent_embeddings, self.neg_t)
            neg_r_e = tf.nn.embedding_lookup(self.rel_embeddings, self.neg_r)

        if config.L1_flag:
            pos = tf.reduce_sum(abs(pos_h_e + pos_r_e - pos_t_e), 1, keep_dims=True)
            neg = tf.reduce_sum(abs(neg_h_e + neg_r_e - neg_t_e), 1, keep_dims=True)
            self.predict = pos
        else:
            pos = tf.reduce_sum((pos_h_e + pos_r_e - pos_t_e) ** 2, 1, keep_dims=True)
            neg = tf.reduce_sum((neg_h_e + neg_r_e - neg_t_e) ** 2, 1, keep_dims=True)
            self.predict = pos

        with tf.name_scope("output"):
            self.loss = tf.reduce_sum(tf.maximum(pos - neg + margin, 0))


def main(_):  # 使用缺省值调用这个函数？但是，函数里怎么使用这个参数呢
    config = Config()
    if (config.testFlag):
        test_lib.init()
        config.relation = test_lib.getRelationTotal()
        config.entity = test_lib.getEntityTotal()
        config.batch = test_lib.getEntityTotal()
        config.batch_size = config.batch
    else:
        lib.init()
        config.relation = lib.getRelationTotal()
        config.entity = lib.getEntityTotal()
        config.batch_size = lib.getTripleTotal() // config.nbatches

    with tf.Graph().as_default():
        sess = tf.Session()
        with sess.as_default():
            initializer = tf.contrib.layers.xavier_initializer(uniform=False)
            with tf.variable_scope("model", reuse=None, initializer=initializer):
                trainModel = TransEModel(config=config)

            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.GradientDescentOptimizer(config.learning_rate)
            grads_and_vars = optimizer.compute_gradients(trainModel.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
            saver = tf.train.Saver()
            sess.run(tf.initialize_all_variables())
            if (config.loadFromData):
                saver.restore(sess, 'model.vec')

            def train_step(pos_h_batch, pos_t_batch, pos_r_batch, neg_h_batch, neg_t_batch, neg_r_batch):
                feed_dict = {
                    trainModel.pos_h: pos_h_batch,
                    trainModel.pos_t: pos_t_batch,
                    trainModel.pos_r: pos_r_batch,
                    trainModel.neg_h: neg_h_batch,
                    trainModel.neg_t: neg_t_batch,
                    trainModel.neg_r: neg_r_batch
                }
                _, step, loss = sess.run(
                    [train_op, global_step, trainModel.loss], feed_dict)
                return loss

            def test_step(pos_h_batch, pos_t_batch, pos_r_batch):
                feed_dict = {
                    trainModel.pos_h: pos_h_batch,
                    trainModel.pos_t: pos_t_batch,
                    trainModel.pos_r: pos_r_batch,
                }
                step, predict = sess.run(
                    [global_step, trainModel.predict], feed_dict)
                return predict

            ph = np.zeros(config.batch_size, dtype=np.int32)
            pt = np.zeros(config.batch_size, dtype=np.int32)
            pr = np.zeros(config.batch_size, dtype=np.int32)
            nh = np.zeros(config.batch_size, dtype=np.int32)
            nt = np.zeros(config.batch_size, dtype=np.int32)
            nr = np.zeros(config.batch_size, dtype=np.int32)

            ph_addr = ph.__array_interface__['data'][0]  # use this interface return a pointer which point to array
            pt_addr = pt.__array_interface__['data'][0]
            pr_addr = pr.__array_interface__['data'][0]
            nh_addr = nh.__array_interface__['data'][0]
            nt_addr = nt.__array_interface__['data'][0]
            nr_addr = nr.__array_interface__['data'][0]

            lib.getBatch.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                     ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
            test_lib.getHeadBatch.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
            test_lib.getTailBatch.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
            test_lib.testHead.argtypes = [ctypes.c_void_p]
            test_lib.testTail.argtypes = [ctypes.c_void_p]

            if not config.testFlag:
                for times in range(config.trainTimes):
                    res = 0.0
                    for batch in range(config.nbatches):
                        lib.getBatch(ph_addr, pt_addr, pr_addr, nh_addr, nt_addr, nr_addr, config.batch_size)
                        res += train_step(ph, pt, pr, nh, nt, nr)
                        current_step = tf.train.global_step(sess, global_step)
                    print(times)
                    print(res)
                saver.save(sess, 'model.vec')
            else:
                total = test_lib.getTestTotal()
                for times in range(total):
                    test_lib.getHeadBatch(ph_addr, pt_addr, pr_addr)
                    res = test_step(ph, pt, pr)
                    test_lib.testHead(res.__array_interface__['data'][0])

                    test_lib.getTailBatch(ph_addr, pt_addr, pr_addr)
                    res = test_step(ph, pt, pr)
                    test_lib.testTail(res.__array_interface__['data'][0])
                    print(times)
                    if (times % 50 == 0):
                        test_lib.test()
                test_lib.test()


if __name__ == "__main__":
    tf.app.run()
