# -*- coding: utf-8 -*-
import tensorflow as tf#.compat.v1
# tf.disable_v2_behavior()
import tensorflow.contrib.slim as slim
# import tf_slim as slim
import numpy as np
import time
import os
from utils import *



class RDN(object):

    def __init__(self,
                 sess,  
                 is_train,
                 is_eval,  
                 image_size, 
                 c_dim,  
                 batch_size,
                 D,  
                 C,  
                 G,
                 P_dim,  
                 Pc_dim,  
                 PD,  
                 PC,  
                 PG,
                 kernel_size  
                 ):

        self.sess = sess
        self.is_train = is_train
        self.is_eval = is_eval
        self.image_size = image_size
        self.c_dim = c_dim
        self.P_dim = P_dim
        self.Pc_dim = Pc_dim
        self.batch_size = batch_size
        self.D = D
        self.C = C
        self.G = G
        self.PD = PD
        self.PC = PC
        self.PG = PG
        self.kernel_size = kernel_size

    def RDBs(self, input_layer):
        rdb_concat = list()
        rdb_in = input_layer
        for i in range(1, self.D + 1):
            x = rdb_in
            for j in range(1, self.C + 1):
                tmp = slim.conv2d(x, self.G, [3, 3], rate=1, activation_fn=lrelu,
                                  weights_initializer=tf.contrib.layers.variance_scaling_initializer())
                x = tf.concat([x, tmp], axis=3)

            # local feature fusion
            x = slim.conv2d(x, self.G, [1, 1], rate=1, activation_fn=None,
                            weights_initializer=tf.contrib.layers.variance_scaling_initializer())
            # local residual learning
            rdb_in = tf.add(x, rdb_in)
            rdb_concat.append(rdb_in)

        return tf.concat(rdb_concat, axis=3)

    def RDBs_S0(self, input_layer):
        r_rdb_concat = list()
        r_rdb_in = input_layer
        for i in range(1, self.PD + 1):
            r_x = r_rdb_in
            for j in range(1, self.PC + 1):
                r_tmp = slim.conv2d(r_x, self.PG, [3, 3], rate=1, activation_fn=lrelu,
                                    weights_initializer=tf.contrib.layers.variance_scaling_initializer())
                r_x = tf.concat([r_x, r_tmp], axis=3)

            # local feature fusion
            r_x = slim.conv2d(r_x, self.PG, [1, 1], rate=1, activation_fn=None,
                              weights_initializer=tf.contrib.layers.variance_scaling_initializer())
            # local residual learning
            r_rdb_in = tf.add(r_x, r_rdb_in)
            r_rdb_concat.append(r_rdb_in)
        return tf.concat(r_rdb_concat, axis=3)

    def RDBs_S1(self, input_layer):
        g_rdb_concat = list()
        g_rdb_in = input_layer
        for i in range(1, self.PD + 1):
            g_x = g_rdb_in
            for j in range(1, self.PC + 1):
                g_tmp = slim.conv2d(g_x, self.PG, [3, 3], rate=1, activation_fn=lrelu,
                                    weights_initializer=tf.contrib.layers.variance_scaling_initializer())
                g_x = tf.concat([g_x, g_tmp], axis=3)

            # local feature fusion
            g_x = slim.conv2d(g_x, self.PG, [1, 1], rate=1, activation_fn=None,
                              weights_initializer=tf.contrib.layers.variance_scaling_initializer())
            # local residual learning
            g_rdb_in = tf.add(g_x, g_rdb_in)
            g_rdb_concat.append(g_rdb_in)
        return tf.concat(g_rdb_concat, axis=3)

    def RDBs_S2(self, input_layer):
        b_rdb_concat = list()
        b_rdb_in = input_layer
        for i in range(1, self.PD + 1):
            b_x = b_rdb_in
            for j in range(1, self.PC + 1):
                b_tmp = slim.conv2d(b_x, self.PG, [3, 3], rate=1, activation_fn=lrelu,
                                    weights_initializer=tf.contrib.layers.variance_scaling_initializer())
                b_x = tf.concat([b_x, b_tmp], axis=3)

            # local feature fusion
            b_x = slim.conv2d(b_x, self.PG, [1, 1], rate=1, activation_fn=None,
                              weights_initializer=tf.contrib.layers.variance_scaling_initializer())
            # local residual learning
            b_rdb_in = tf.add(b_x, b_rdb_in)
            b_rdb_concat.append(b_rdb_in)
        return tf.concat(b_rdb_concat, axis=3)

    # def upsample(self,x1,  output_channels, in_channels, scope_name):
    #     with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE) as scope:
    #         pool_size = 2
    #         deconv_filter = tf.get_variable('weights', [pool_size, pool_size, output_channels, in_channels],
    #                                         trainable=True)
    #         if self.is_train:
    #             deconv = tf.nn.conv2d_transpose(x1, deconv_filter, [self.batch_size,int(self.images.shape[1]),int(self.images.shape[2]),in_channels], strides=[1, pool_size, pool_size, 1],
    #                                         name=scope_name)#tf.shape(x2)
    #         if not self.is_train:
    #             deconv = tf.nn.conv2d_transpose(x1, deconv_filter, [1,int(self.images.shape[1]),int(self.images.shape[2]),in_channels], strides=[1, pool_size, pool_size, 1],
    #                                         name=scope_name)#
    #         # deconv_output = tf.concat([deconv, x2], 3)
    #         deconv.set_shape([None, None, None, output_channels])
    #
    #         return deconv
    def model(self):
        image_0 = self.images[:, 1:self.images.shape[1]:2, 1:self.images.shape[2]:2,:]
        image_45 = self.images[:, 0:self.images.shape[1]:2, 1:self.images.shape[2]:2,:]
        image_90 = self.images[:, 0:self.images.shape[1]:2, 0:self.images.shape[2]:2,:]
        image_135 = self.images[:, 1:self.images.shape[1]:2, 0:self.images.shape[2]:2,:]


        gt_0 = self.labels[:, 1:self.labels.shape[1]:2, 1:self.labels.shape[2]:2,:]
        gt_45 = self.labels[:, 0:self.labels.shape[1]:2, 1:self.labels.shape[2]:2, :]
        gt_90 = self.labels[:, 0:self.labels.shape[1]:2, 0:self.labels.shape[2]:2, :]
        gt_135 = self.labels[:, 1:self.labels.shape[1]:2, 0:self.labels.shape[2]:2, :]


        F_1_0 = slim.conv2d(image_0, self.G, [3, 3], rate=1, activation_fn=lrelu,
                          weights_initializer=tf.contrib.layers.variance_scaling_initializer())
        F0_0 = slim.conv2d(F_1_0, self.G, [3, 3], rate=1, activation_fn=lrelu,
                         weights_initializer=tf.contrib.layers.variance_scaling_initializer())

        FD_0 = self.RDBs(F0_0)

        FGF1_0 = slim.conv2d(FD_0, self.G, [1, 1], rate=1, activation_fn=None,
                           weights_initializer=tf.contrib.layers.variance_scaling_initializer())
        FGF2_0 = slim.conv2d(FGF1_0, self.G, [3, 3], rate=1, activation_fn=lrelu,
                           weights_initializer=tf.contrib.layers.variance_scaling_initializer())

        FDF_0 = tf.add(FGF2_0, F_1_0)

        IHR_0 = slim.conv2d(FDF_0, self.c_dim, [3, 3], rate=1, activation_fn=lrelu,
                          weights_initializer=tf.contrib.layers.variance_scaling_initializer())
        PSNR_0 = psnr(IHR_0, gt_0)


        F_1_45 = slim.conv2d(image_45, self.G, [3, 3], rate=1, activation_fn=lrelu,
                          weights_initializer=tf.contrib.layers.variance_scaling_initializer())
        F0_45 = slim.conv2d(F_1_45, self.G, [3, 3], rate=1, activation_fn=lrelu,
                         weights_initializer=tf.contrib.layers.variance_scaling_initializer())

        FD_45 = self.RDBs(F0_45)

        FGF1_45 = slim.conv2d(FD_45, self.G, [1, 1], rate=1, activation_fn=None,
                           weights_initializer=tf.contrib.layers.variance_scaling_initializer())
        FGF2_45 = slim.conv2d(FGF1_45, self.G, [3, 3], rate=1, activation_fn=lrelu,
                           weights_initializer=tf.contrib.layers.variance_scaling_initializer())

        FDF_45 = tf.add(FGF2_45, F_1_45)

        IHR_45 = slim.conv2d(FDF_45, self.c_dim, [3, 3], rate=1, activation_fn=lrelu,
                          weights_initializer=tf.contrib.layers.variance_scaling_initializer())
        PSNR_45 = psnr(IHR_45, gt_45)


        F_1_90 = slim.conv2d(image_90, self.G, [3, 3], rate=1, activation_fn=lrelu,
                          weights_initializer=tf.contrib.layers.variance_scaling_initializer())
        F0_90 = slim.conv2d(F_1_90, self.G, [3, 3], rate=1, activation_fn=lrelu,
                         weights_initializer=tf.contrib.layers.variance_scaling_initializer())

        FD_90 = self.RDBs(F0_90)

        FGF1_90 = slim.conv2d(FD_90, self.G, [1, 1], rate=1, activation_fn=None,
                           weights_initializer=tf.contrib.layers.variance_scaling_initializer())
        FGF2_90 = slim.conv2d(FGF1_90, self.G, [3, 3], rate=1, activation_fn=lrelu,
                           weights_initializer=tf.contrib.layers.variance_scaling_initializer())

        FDF_90 = tf.add(FGF2_90, F_1_90)

        IHR_90 = slim.conv2d(FDF_90, self.c_dim, [3, 3], rate=1, activation_fn=lrelu,
                          weights_initializer=tf.contrib.layers.variance_scaling_initializer())
        PSNR_90 = psnr(IHR_90, gt_90)


        F_1_135 = slim.conv2d(image_135, self.G, [3, 3], rate=1, activation_fn=lrelu,
                          weights_initializer=tf.contrib.layers.variance_scaling_initializer())
        F0_135 = slim.conv2d(F_1_135, self.G, [3, 3], rate=1, activation_fn=lrelu,
                         weights_initializer=tf.contrib.layers.variance_scaling_initializer())

        FD_135 = self.RDBs(F0_135)

        FGF1_135 = slim.conv2d(FD_135, self.G, [1, 1], rate=1, activation_fn=None,
                           weights_initializer=tf.contrib.layers.variance_scaling_initializer())
        FGF2_135 = slim.conv2d(FGF1_135, self.G, [3, 3], rate=1, activation_fn=lrelu,
                           weights_initializer=tf.contrib.layers.variance_scaling_initializer())

        FDF_135 = tf.add(FGF2_135, F_1_135)

        IHR_135 = slim.conv2d(FDF_135, self.c_dim, [3, 3], rate=1, activation_fn=lrelu,
                          weights_initializer=tf.contrib.layers.variance_scaling_initializer())

        PSNR_135 = psnr(IHR_135, gt_135)


        p_s0 = tf.concat([IHR_0,IHR_45,IHR_90,IHR_135],axis=-1)
        p_s1 = tf.concat([IHR_0,IHR_90],axis=-1)
        p_s2 = tf.concat([IHR_45,IHR_135],axis=-1)

        gt_s0 = (gt_0 + gt_45 + gt_90 + gt_135)/2
        gt_s1 = gt_0 - gt_90
        gt_s2 = gt_45 - gt_135


        # S0
        RF_0 = slim.conv2d(p_s0, self.Pc_dim, [3, 3], rate=1, activation_fn = None, weights_initializer=tf.contrib.layers.variance_scaling_initializer())
        RF_1 = slim.conv2d(RF_0, self.PG, [3, 3], rate=1, activation_fn=lrelu,
                           weights_initializer=tf.contrib.layers.variance_scaling_initializer())
        RF0 = slim.conv2d(RF_1, self.PG, [3, 3], rate=1, activation_fn=lrelu,
                          weights_initializer=tf.contrib.layers.variance_scaling_initializer())

        RFD = self.RDBs_S0(RF0)
        RFGF1 = slim.conv2d(RFD, self.PG, [1, 1], rate=1, activation_fn=None,
                            weights_initializer=tf.contrib.layers.variance_scaling_initializer())
        RFGF2 = slim.conv2d(RFGF1, self.PG, [3, 3], rate=1, activation_fn=lrelu,
                            weights_initializer=tf.contrib.layers.variance_scaling_initializer())
        RIHR1 = tf.add(RFGF2,RF_1)
        RIHR_s0 = slim.conv2d(RIHR1, self.Pc_dim, [3, 3], rate=1, activation_fn=lrelu,
                           weights_initializer=tf.contrib.layers.variance_scaling_initializer())



        PSNR_s0 = psnr(RIHR_s0/2, gt_s0/2)


        # s1
        GF_0 = slim.conv2d(p_s1, self.Pc_dim, [3, 3], rate=1, activation_fn = None, weights_initializer=tf.contrib.layers.variance_scaling_initializer())
        GF_1 = slim.conv2d(GF_0, self.PG, [3, 3], rate=1, activation_fn=lrelu,
                           weights_initializer=tf.contrib.layers.variance_scaling_initializer())
        GF0 = slim.conv2d(GF_1, self.PG, [3, 3], rate=1, activation_fn=lrelu,
                          weights_initializer=tf.contrib.layers.variance_scaling_initializer())
        GFD = self.RDBs_S1(GF0)

        GFGF1 = slim.conv2d(GFD, self.PG, [1, 1], rate=1, activation_fn=None,
                            weights_initializer=tf.contrib.layers.variance_scaling_initializer())
        GFGF2 = slim.conv2d(GFGF1, self.PG, [3, 3], rate=1, activation_fn=lrelu,
                            weights_initializer=tf.contrib.layers.variance_scaling_initializer())
        GIHR1 = tf.add(GFGF2,GF_1)
        GIHR_s1 = slim.conv2d(GIHR1, self.Pc_dim, [3, 3], rate=1, activation_fn=lrelu,
                           weights_initializer=tf.contrib.layers.variance_scaling_initializer())

        PSNR_s1 = psnr((GIHR_s1+1)/2, (gt_s1+1)/2)



        # s2
        BF_0 = slim.conv2d(p_s2, self.Pc_dim, [3, 3], rate=1, activation_fn = None, weights_initializer=tf.contrib.layers.variance_scaling_initializer())
        BF_1 = slim.conv2d(BF_0, self.PG, [3, 3], rate=1, activation_fn=lrelu,
                           weights_initializer=tf.contrib.layers.variance_scaling_initializer())
        BF0 = slim.conv2d(BF_1, self.PG, [3, 3], rate=1, activation_fn=lrelu,
                          weights_initializer=tf.contrib.layers.variance_scaling_initializer())

        BFD = self.RDBs_S2(BF0)

        BFGF1 = slim.conv2d(BFD, self.PG, [1, 1], rate=1, activation_fn=None,
                            weights_initializer=tf.contrib.layers.variance_scaling_initializer())
        BFGF2 = slim.conv2d(BFGF1, self.PG, [3, 3], rate=1, activation_fn=lrelu,
                            weights_initializer=tf.contrib.layers.variance_scaling_initializer())
        BIHR1 = tf.add(BFGF2,BF_1)

        BIHR_s2 = slim.conv2d(BIHR1, self.Pc_dim, [3, 3], rate=1, activation_fn=lrelu,
                           weights_initializer=tf.contrib.layers.variance_scaling_initializer())
        PSNR_s2 = psnr((BIHR_s2+1)/2, (gt_s2+1)/2)



        PRED_DOLP = tf.div(tf.sqrt(tf.square(GIHR_s1) + tf.square(BIHR_s2)), (RIHR_s0 + 0.00001))
        PRED_DOLP = tf.clip_by_value(PRED_DOLP, 0, 1)
        PRED_AOP = 1 / 2 * tf.atan2(BIHR_s2, GIHR_s1)
        PRED_AOP = (PRED_AOP + math.pi / 2.0) / (math.pi)


        GT_DOLP = tf.div(tf.sqrt(tf.square(gt_s1) + tf.square(gt_s2)), (gt_s0 + 0.00001))
        GT_DOLP = tf.clip_by_value(GT_DOLP, 0, 1)
        GT_AOP = 1 / 2 * tf.atan2(gt_s2, gt_s1)
        GT_AOP = (GT_AOP + math.pi / 2.0) / (math.pi)

        PSNR_aop = psnr(PRED_AOP, GT_AOP)
        PSNR_dolp = psnr(PRED_DOLP,GT_DOLP)

        return IHR_0, IHR_45, IHR_90, IHR_135, gt_0, gt_45, gt_90, gt_135, RIHR_s0, GIHR_s1, BIHR_s2, gt_s0, gt_s1, gt_s2,PRED_AOP,PRED_DOLP,GT_AOP,GT_DOLP, PSNR_0, PSNR_45,PSNR_90,PSNR_135,PSNR_s0,PSNR_s1,PSNR_s2,PSNR_aop,PSNR_dolp


    def build_model(self, images_shape, labels_shape):
        self.images = tf.placeholder(tf.float32, images_shape, name='images')
        self.labels = tf.placeholder(tf.float32, labels_shape, name='labels')

        self.pred_0, self.pred_45, self.pred_90, self.pred_135, self.gt_0, self.gt_45, self.gt_90,self.gt_135, self.pred_s0, self.pred_s1, self.pred_s2, self.gt_s0, self.gt_s1, self.gt_s2,self.pred_aop,self.pred_dolp,self.gt_aop,self.gt_dolp, self.psnr_0, self.psnr_45, self.psnr_90, self.psnr_135, self.psnr_s0, self.psnr_s1,self.psnr_s2,self.psnr_aop,self.psnr_dolp = self.model()


        # self.loss_0 = tf.reduce_mean(tf.abs(self.gt_0 - self.pred_0))+0.0000001
        # self.loss_45 = tf.reduce_mean(tf.abs(self.gt_45 - self.pred_45))+0.0000001
        # self.loss_90 = tf.reduce_mean(tf.abs(self.gt_90 - self.pred_90))+0.0000001
        # self.loss_135 = tf.reduce_mean(tf.abs(self.gt_135 - self.pred_135))+0.0000001
        # w0 = self.loss_0/(self.loss_0 + self.loss_45 + self.loss_90 + self.loss_135)
        # w45 = self.loss_45/(self.loss_0 + self.loss_45 + self.loss_90 + self.loss_135)
        # w90 = self.loss_90/(self.loss_0 + self.loss_45 + self.loss_90 + self.loss_135)
        # w135 = self.loss_135/(self.loss_0 + self.loss_45 + self.loss_90 + self.loss_135)
        # self.n_loss = w0*self.loss_0 + w45*self.loss_45 + w90*self.loss_90 + w135*self.loss_135



        self.s0_loss = tf.reduce_mean(tf.abs(self.gt_s0 - self.pred_s0))+0.0000001
        self.s1_loss = tf.reduce_mean(tf.abs(self.gt_s1 - self.pred_s1))+0.0000001
        self.s2_loss = tf.reduce_mean(tf.abs(self.gt_s2 - self.pred_s2))+0.0000001

        self.dolp_loss = 5*tf.reduce_mean(tf.abs(self.gt_dolp - self.pred_dolp)) + 0.0000001  
        self.aop_loss = 2*tf.reduce_mean(tf.abs(self.gt_aop - self.pred_aop)) + 0.0000001  
        self.S0_loss = tf.reduce_mean(tf.abs(self.gt_s0 - self.pred_s0)) + 0.0000001

        wS = self.S0_loss / (self.aop_loss + self.dolp_loss + self.S0_loss)
        wd = self.dolp_loss / (self.aop_loss + self.dolp_loss + self.S0_loss)
        wa = self.aop_loss / (self.aop_loss + self.dolp_loss + self.S0_loss)
        self.Spad_loss = wd * self.dolp_loss + wa * self.aop_loss + wS*self.S0_loss

        s0dy_true, s0dx_true = tf.image.image_gradients(self.gt_s0)
        s0dy_pred, s0dx_pred = tf.image.image_gradients(self.pred_s0)
        s1dy_true, s1dx_true = tf.image.image_gradients(self.gt_s1)
        s1dy_pred, s1dx_pred = tf.image.image_gradients(self.pred_s1)
        s2dy_true, s2dx_true = tf.image.image_gradients(self.gt_s2)
        s2dy_pred, s2dx_pred = tf.image.image_gradients(self.pred_s2)
        self.grads0loss = self.s0_loss + 0.5*tf.reduce_mean(tf.abs(s0dy_pred - s0dy_true) + tf.abs(s0dx_pred - s0dx_true))+0.0000001
        self.grads1loss = self.s1_loss + 0.5*tf.reduce_mean(tf.abs(s1dy_pred - s1dy_true) + tf.abs(s1dx_pred - s1dx_true))+0.0000001
        self.grads2loss = self.s2_loss + 0.5*tf.reduce_mean(tf.abs(s2dy_pred - s2dy_true) + tf.abs(s2dx_pred - s2dx_true))+0.0000001
        #
        # self.gradloss = ( self.grads1loss +  self.grads2loss + self.grads0loss)*0.5+0.0000001

        wgs0 = self.grads0loss / (self.grads0loss + self.grads1loss + self.grads2loss)
        wgs1 = self.grads1loss / (self.grads0loss + self.grads1loss + self.grads2loss)
        wgs2 = self.grads2loss / (self.grads0loss + self.grads1loss + self.grads2loss)
        self.gradloss = wgs0 * self.grads0loss +wgs1 * self.grads1loss + wgs2 * self.grads2loss 

        self.loss = 0.7*self.gradloss + 0.3*self.Spad_loss

        tf.summary.scalar('loss', self.loss)
        # tf.summary.scalar('closs', self.n_loss)
        tf.summary.scalar('PSNR_s0', self.psnr_s0)
        tf.summary.scalar('PSNR_S1', self.psnr_s1)
        tf.summary.scalar('PSNR_s2', self.psnr_s2)
        tf.summary.scalar('PSNR_aop', self.psnr_aop)
        tf.summary.scalar('PSNR_dolp', self.psnr_dolp)

        self.saver = tf.train.Saver(max_to_keep=5)

    def train(self, config):

        input_setup(config) 

        train_data_dir, eval_data_dir = get_data_dir(config.is_train)
        train_data_num = get_data_num(train_data_dir)
        batch_num = train_data_num // config.batch_size
        eval_data_num = get_data_num(eval_data_dir)


        images_shape = [None, self.image_size, self.image_size , self.c_dim]
        labels_shape = [None, self.image_size, self.image_size, self.c_dim]

        self.build_model(images_shape, labels_shape)

        epoch, counter = self.load(config.checkpoint_dir)
        global_step = tf.Variable(counter, trainable=False)
        learning_rate = tf.train.exponential_decay(config.learning_rate, global_step, config.lr_decay_steps * batch_num,
                                                   config.lr_decay_rate, staircase=True)
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        learning_step = optimizer.minimize(self.loss, global_step=global_step)

        tf.global_variables_initializer().run(session=self.sess)

        merged_summary_op = tf.summary.merge_all()
        summary_train_path = os.path.join(config.checkpoint_dir, "train_%s_%s_%s" % (self.D, self.C, self.G))
        summary_eval_path = os.path.join(config.checkpoint_dir, "eval_%s_%s_%s" % (self.D, self.C, self.G))

        summary_writer_train = tf.summary.FileWriter(summary_train_path, self.sess.graph)
        summary_writer_validate = tf.summary.FileWriter(summary_eval_path)

        time_all = time.time()
        print("\nNow Start Training...\n")
        model_dir = "%s_%s_%s_%s" % ("rdn", self.D, self.C, self.G)
        checkpoint_dir = os.path.join(config.checkpoint_dir, model_dir)
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)

        if ckpt and ckpt.model_checkpoint_path:
            ckpt_path = str(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(os.getcwd(), ckpt_path))

        for ep in range(epoch, config.epoch):
            # Run by batch images

            for idx in range(0, batch_num):
                batch_images, batch_labels = get_batch(train_data_dir, train_data_num, config.batch_size)
                eval_batch_images, eval_batch_labels = get_batch(eval_data_dir, eval_data_num, config.batch_size)
                counter += 1

                assert batch_images.shape == batch_labels.shape
                assert eval_batch_images.shape == eval_batch_labels.shape
                _, loss, lr, psnr_0,psnr_45, psnr_90, psnr_135, psnr_s0,psnr_s1,psnr_s2,psnr_aop,psnr_dolp = self.sess.run(
                    [learning_step, self.loss, learning_rate, self.psnr_0, self.psnr_45,
                     self.psnr_90, self.psnr_135,self.psnr_s0,self.psnr_s1,self.psnr_s2,self.psnr_aop,self.psnr_dolp],
                    feed_dict={self.images: batch_images, self.labels: batch_labels})
                eval_loss = self.sess.run(self.loss,
                                          feed_dict={self.images: eval_batch_images,
                                                     self.labels: eval_batch_labels})
                if counter % 10 == 0:
                    print(
                        "Epoch: [%2d], batch: [%2d/%2d], step: [%2d], time: [%d]min, psnr_0:[%2.2f], psnr_45:[%2.2f], psnr_90:[%2.2f], psnr_135:[%2.2f],psnr_s0:[%2.2f],psnr_s1:[%2.2f],psnr_s2:[%2.2f],psnr_aop:[%2.2f],psnr_dolp:[%2.2f], train_loss: [%.4f], eval_loss:[%.4f]" % (
                            ep + 1, idx, batch_num, counter, int((time.time()-time_all)/60), psnr_0, psnr_45, psnr_90,psnr_135,psnr_s0,psnr_s1,psnr_s2,psnr_aop,psnr_dolp,
                             loss, eval_loss))

                if counter % 100 == 0:
                    print(int((time.time() - time_all) / 60))
                    self.save(config.checkpoint_dir, ep + 1, counter)
                    summary_train = self.sess.run(merged_summary_op,
                                                  feed_dict={self.images: batch_images, self.labels: batch_labels})
                    summary_writer_train.add_summary(summary_train, counter)
                    summary_eval = self.sess.run(merged_summary_op,
                                                 feed_dict={self.images: eval_batch_images,
                                                            self.labels: eval_batch_labels})
                    summary_writer_validate.add_summary(summary_eval, counter)
                if counter > 0 and counter == batch_num * config.epoch:
                    print("Congratulation !  Train Finished.")
                    print("Congratulation !  Train Finished.")
                    print("Congratulation !  Train Finished.")
                    return

    def test(self, config):
        print("\nPrepare Testing Data...\n")
        paths = prepare_data(config)  
        data_num = len(paths)

        print("\nNow Start Testing...\n")
        times = []
        s = time.time()
        for idx in range(data_num):
            output_name = paths[idx].split("/")[-1].split('.')[0]
            ratio = float(30)
            input_ = cv2.imread(paths[idx], -1)
            input_ = input_ * ratio
            input_ = input_[np.newaxis, :]

            test_ = np.zeros([input_.shape[0], input_.shape[1], input_.shape[2], input_.shape[3]])
            images_shape = input_.shape
            labels_shape = test_.shape

            self.build_model(images_shape, labels_shape)
            tf.global_variables_initializer().run(session=self.sess)

            self.load(config.checkpoint_dir)
            s = time.time()
            result_s0, result_s1, result_s2 = self.sess.run([self.pred_s0, self.pred_s1, self.pred_s2],
                                                              feed_dict={self.images: input_/ 255})

            tot = time.time() - s
            self.sess.close()
            tf.reset_default_graph()
            self.sess = tf.Session()


            times.append(tot)

            img_s0 = np.squeeze(result_s0)
            img_s1 = np.squeeze(result_s1)
            img_s2 = np.squeeze(result_s2)
            save_our(config,img_s0,img_s1,img_s2,output_name)


        print("\n All Done ! ")
        print("\nTotal images: {0}".format(data_num))
        Ttime, Mtime = np.sum(times[1:]), np.mean(times[1:])
        print("Time taken: {0} sec at {1} fps".format(Ttime, 1./Mtime))

    def load(self, checkpoint_dir):
        print("\nReading Checkpoints.....\n")
        model_dir = "%s_%s_%s_%s" % ("rdn", self.D, self.C, self.G)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)

        if ckpt and ckpt.model_checkpoint_path:
            ckpt_path = str(ckpt.model_checkpoint_path)
            print(os.path.join(os.getcwd(), ckpt_path))
            self.saver.restore(self.sess, os.path.join(os.getcwd(), ckpt_path))
            step = int(ckpt_path.split('-')[-1])
            epoch = int(ckpt_path.split('-')[1])
            print("\nCheckpoint Loading Success! %s\n" % ckpt_path)
        else:
            step = 0
            epoch = 0
            print("\nCheckpoint Loading Failed! \n")

        return epoch, step

    def save(self, checkpoint_dir, epoch, step):
        model_name = "RDN.model"
        model_dir = "%s_%s_%s_%s" % ("rdn", self.D, self.C, self.G)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        save_name = os.path.join(checkpoint_dir, model_name + '-{}'.format(epoch + 1))
        self.saver.save(self.sess,
                        save_name,
                        global_step=step)
