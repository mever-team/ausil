import numpy as np
import tensorflow as tf
tf.disable_v2_behavior()
import time

usegpu = True


class PCA_layer(object):

    def __init__(self, dims=2528):

        with tf.variable_scope('PCA'):
            self.mean = tf.get_variable('mean_sift', dtype=tf.float32, trainable=False, shape=(dims,) )

            self.weights = tf.get_variable('weights', dtype=tf.float32, trainable=False, shape=(dims,dims))

    def __call__(self, logits):
        logits = logits - self.mean
        logits = tf.tensordot(logits, self.weights, axes=1)
        return logits


class Attention_layer(object):

    def __init__(self, dims=2528):
        with tf.variable_scope('attention_layer'):
            self.context_vector = tf.get_variable('context_vector', dtype=tf.float32,
                                                  trainable=False, shape=(dims, 1))

    def __call__(self, logits):
        weights = tf.tensordot(logits, self.context_vector, axes=1) / 2.0 + 0.5
        return tf.multiply(logits, weights), weights


class Video_Comparator(object):

    def __init__(self):

        self.conv1 = tf.keras.layers.Conv2D(32, [3, 3], activation='relu')
        self.mpool1 = tf.keras.layers.MaxPool2D([2, 2], 2)
        self.conv2 = tf.keras.layers.Conv2D(64, [3, 3], activation='relu')
        self.mpool2 = tf.keras.layers.MaxPool2D([2, 2], 2)
        self.conv3 = tf.keras.layers.Conv2D(128, [3, 3], activation='relu')
        self.fconv = tf.keras.layers.Conv2D(1, [1, 1])

    def __call__(self, sim_matrix):
        with tf.variable_scope('video_comparator'):
            sim = tf.reshape(sim_matrix, (1, tf.shape(sim_matrix)[0], tf.shape(sim_matrix)[1], 1))
            sim = tf.pad(sim, [[0, 0], [1, 1], [1, 1], [0, 0]], 'SYMMETRIC')
            sim = self.conv1(sim)
            sim = self.mpool1(sim)
            sim = tf.pad(sim, [[0, 0], [1, 1], [1, 1], [0, 0]], 'SYMMETRIC')
            sim = self.conv2(sim)
            sim = self.mpool2(sim)
            sim = tf.pad(sim, [[0, 0], [1, 1], [1, 1], [0, 0]], 'SYMMETRIC')
            sim = self.conv3(sim)
            sim = self.fconv(sim)

            sim = tf.squeeze(sim, [0, 3])

            sim = tf.clip_by_value(sim, -1, 1)  # Hard tanh

        return sim


class AuSiL(object):

    def __init__(self, model_dir, load_queries=False, queries_number=None, gpu_id=0):

        with tf.device('/gpu:%s' % gpu_id):

            self.load_queries = load_queries

            self.pca_layer = PCA_layer()
            self.att_layer = Attention_layer()
            self.vid_comp = Video_Comparator()

            self.frames = tf.placeholder(tf.float32, [None, 2528], name='frames')
            self.embeddings = self.extract_features(self.frames)

            if self.load_queries:   # Load queries on GPU memory.
                self.queries = [tf.Variable( np.zeros( (1,2528) ), dtype=tf.float32, validate_shape=False)
                              for _ in range(queries_number)]
                self.candidate = tf.placeholder(tf.float32, [None, None], name='candidate')

                self.similarities = []
                for q in self.queries:
                    sim = tf.matmul(q, tf.transpose(self.candidate))    # Sim Matrix
                    sim = self.vid_comp(sim)
                    sim = self.chamfer_similarity(sim)
                    self.similarities.append(sim)

            else:   # Do NOT load queries on GPU memory.
                self.query = tf.placeholder(tf.float32, [None, None], name= 'query')
                self.candidate = tf.placeholder(tf.float32, [None, None], name='candidate')

                sim = tf.matmul(self.query, tf.transpose(self.candidate))   # Similarity Matrix
                self.before = sim  # Similarity Matrix

                sim = self.vid_comp(sim)
                self.after = sim  # CNN output

                self.similarity = self.chamfer_similarity(sim)  # Overall Similarity

        # Without this (next 2 lines), ERROR occurs
        x = tf.Variable(tf.zeros([100, 100]))  # Without this, ERROR occurs
        self.vid_comp(x)

        init = self.load_model(model_dir)
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True

        self.sess = tf.Session(config=config)
        self.sess.run(init)

    def extract_features(self, features):
        features = tf.nn.l2_normalize(features, -1, epsilon=1e-15)
        features = self.pca_layer(features)
        features = tf.nn.l2_normalize(features, -1, epsilon=1e-15)
        features, weights = self.att_layer(features)
        return features

    def get_features(self, frames):
        features = self.sess.run(self.embeddings, feed_dict={self.frames: frames})
        return features

    def chamfer_similarity(self, sim, max_axis=1, mean_axis=0):
        sim = tf.reduce_max(sim, axis=max_axis, keepdims=True)
        sim = tf.reduce_mean(sim, axis=mean_axis, keepdims=True)
        return tf.squeeze(sim, [max_axis, mean_axis])

    def calculate_sim(self, candidate):
        candidate = self.sess.run(self.embeddings, feed_dict={self.frames: candidate})
        if self.load_queries:
            sim = self.sess.run(self.similarities, feed_dict={self.candidate: candidate})
        else:
            sim = [self.calculate_sim_single(q, candidate) for q in self.queries]
        return sim

    def calculate_sim_single(self, query, candidate):
        return self.sess.run(self.similarity, feed_dict={self.query:query, self.candidate:candidate})

    def calculate_sim_one_to_one(self, query, candidate):
        query = self.sess.run(self.embeddings, feed_dict={self.frames: query})
        candidate = self.sess.run(self.embeddings, feed_dict={self.frames: candidate})
        before = self.sess.run(self.before, feed_dict={self.query: query, self.candidate: candidate})
        after = self.sess.run(self.after, feed_dict={self.query: query, self.candidate: candidate})
        start = time.time()
        sim = self.sess.run(self.similarity, feed_dict={self.query: query, self.candidate: candidate})
        end = time.time()
        timer = end-start
        return sim, timer, before, after

    def load_model(self, model_path):
        previous_variables = [var_name for var_name, _ in tf.contrib.framework.list_variables(model_path)]
        restore_map = {variable.op.name: variable for variable in tf.global_variables()
                       if variable.op.name in previous_variables}
        print('{} layers loaded'.format(len(restore_map)))
        #print(tf.contrib.framework.list_variables(model_path))
        print(restore_map)
        #print(tf.global_variables())
        tf.contrib.framework.init_from_checkpoint(model_path, restore_map)
        tf_init = tf.global_variables_initializer()
        return tf_init

    def set_queries(self, queries):
        if self.load_queries:
            for i in range(len(queries)):
                self.sess.run(tf.assign(self.queries[i], queries[i], validate_shape=False))
        else:
            self.queries = queries
