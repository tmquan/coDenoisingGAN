from Utility import *
from GAN import *

###############################################################################


class Model(GANModelDesc):
    # def build_losses(self, logits_real, logits_fake, name="GAN_loss"):
    #   with tf.name_scope(name=name):
    #       score_real = tf.sigmoid(logits_real)
    #       score_fake = tf.sigmoid(logits_fake)
    #       tf.summary.histogram('score-real', score_real)
    #       tf.summary.histogram('score-fake', score_fake)
    #       with tf.name_scope("discrim"):
    #           d_loss_pos = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
    #               logits=logits_real, labels=tf.ones_like(logits_real)), name='loss_real')
    #           d_loss_neg = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
    #               logits=logits_fake, labels=tf.zeros_like(logits_fake)), name='loss_fake')

    #           d_pos_acc = tf.reduce_mean(tf.cast(score_real > 0.5, tf.float32), name='accuracy_real')
    #           d_neg_acc = tf.reduce_mean(tf.cast(score_fake < 0.5, tf.float32), name='accuracy_fake')

    #           d_accuracy = tf.add(.5 * d_pos_acc, .5 * d_neg_acc, name='accuracy')
    #           d_loss = tf.add(.5 * d_loss_pos, .5 * d_loss_neg, name='loss')
    #       with tf.name_scope("gen"):
    #           g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
    #               logits=logits_fake, labels=tf.ones_like(logits_fake)), name='loss')
    #           g_accuracy = tf.reduce_mean(tf.cast(score_fake > 0.5, tf.float32), name='accuracy')
    #           return g_loss, d_loss

    def build_losses(self, vecpos, vecneg, name="WGAN_loss"):
        with tf.name_scope(name=name):
            # the Wasserstein-GAN losses
            d_loss = tf.reduce_mean(vecneg - vecpos, name='d_loss')
            g_loss = tf.negative(tf.reduce_mean(vecneg), name='g_loss')
            # add_moving_summary(self.d_loss, self.g_loss)
            return g_loss, d_loss

    # def build_losses(self, real, fake, name="LSGAN_loss"):
    #   d_real = tf.reduce_mean(tf.squared_difference(real, 1), name='d_real')
    #   d_fake = tf.reduce_mean(tf.square(fake), name='d_fake')
    #   d_loss = tf.multiply(d_real + d_fake, 0.5, name='d_loss')
    #   tf.summary.histogram('score-real', d_real)
    #   tf.summary.histogram('score-fake', d_fake)
    #   g_loss = tf.reduce_mean(tf.squared_difference(fake, 1), name='g_loss')
    #   # add_moving_summary(g_loss, d_loss)
    #   return g_loss, d_loss

    @auto_reuse_variable_scope
    def encoder(self, image):
        assert image is not None
        enc, feats = arch_fusionnet_encoder_2d(tf_2tanh(image, maxVal=1.0), nb_filters=16)
        return enc, feats
    
    @auto_reuse_variable_scope
    def decoder(self, image, feats=[None, None, None]):
        assert image is not None
        dec, feats =  arch_fusionnet_decoder_2d(image, feats, nb_filters=16)
        return tf_2imag(dec, maxVal=1.0), feats 

    @auto_reuse_variable_scope
    def generator(self, image):
        assert image is not None
        # with tf.variable_scope('pack')
        enc, _ = self.encoder(image)
        dec, _ = self.decoder(enc)
        return dec

    @auto_reuse_variable_scope
    def discriminator(self, image):
        assert image is not None
        # with tf.variable_scope('pack')
        enc, _ = self.encoder(image)
        return enc


    def inputs(self):
        return [
            tf.placeholder(tf.float32, (DIMZ, DIMY, DIMX, 1), 'image'),
            tf.placeholder(tf.float32, (DIMZ, DIMY, DIMX, 1), 'label'),
            tf.placeholder(tf.float32, (DIMZ, DIMY, DIMX, 1), 'noise'),
            ]

    def build_graph(self, image, label, noise):
        print(image)
        print(label)
        print(noise)
        pi, pl, pn = image, label, noise

    

        with tf.variable_scope('gen'):
            with tf.variable_scope('pre'): # Preprocessing
                # Scale everything in between 0 and 1
                pi = pi / 255.0 # noise ppc
                pl = pl / 255.0 # clean wfly
                pn = pn / 255.0 # noise true

                # Multiply the noise to label (wfly clean) to make input (wfly_noise)
                pnl = pn * pl
                pni = pi
            with tf.variable_scope('enc0'): # wfly
                e1, _ = self.encoder(pnl)
            with tf.variable_scope('dec0'): # wfly
                n1, _       = self.decoder(e1)

            with tf.variable_scope('enc0'): # ppc
                e2, _ = self.encoder(pi)
            with tf.variable_scope('dec0'): # ppc
                n2, _       = self.decoder(e2)

            with tf.variable_scope('pos'): # Postprocessing
                # with varreplace.freeze_variables():
                eps = 1e-9
                c1 = (pnl) / (n1 + eps) # wfly
                c2 = (pni) / (n2 + eps) # ppc

        with tf.variable_scope('discrim'):
            with tf.variable_scope('wfly'):
                real_wfly1  , _ = self.encoder(pl)
                fake_wfly1_1, _ = self.encoder(c1)
                fake_wfly1_2, _ = self.encoder(c2)
            with tf.variable_scope('noise'):
                real_noise_0, _ = self.encoder(pn)
                fake_noise_1, _ = self.encoder(n1)
                fake_noise_2, _ = self.encoder(n2)

        ######################################
        # Loss computation
        g_losses = []
        d_losses = []

        # MSE loss
        with tf.name_scope('loss_mae'):
            mae_wfly1 = tf.reduce_mean(tf.abs(pl - c1), name='mae_wfly1')
            g_losses.append(1e1*mae_wfly1)
            add_moving_summary(mae_wfly1)  

            mae_noise = tf.reduce_mean(tf.abs(pn - n1), name='mae_noise')
            g_losses.append(1e1*mae_noise)
            add_moving_summary(mae_noise)   

 

        with tf.name_scope('loss_gan'):
            G_loss_1, D_loss_1 = self.build_losses(real_wfly1, fake_wfly1_1, name='w_wfly1')
            g_losses.append(1e0*G_loss_1)
            d_losses.append(1e0*D_loss_1)
            
            G_loss_2, D_loss_2 = self.build_losses(real_wfly1, fake_wfly1_2, name='w_ppc')
            g_losses.append(1e0*G_loss_2)
            d_losses.append(1e0*D_loss_2)

            G_loss_3, D_loss_3 = self.build_losses(real_noise_0, fake_noise_1, name='n_wfly1')
            g_losses.append(1e0*G_loss_3)
            d_losses.append(1e0*D_loss_3)
            
            G_loss_4, D_loss_4 = self.build_losses(real_noise_0, fake_noise_2, name='n_ppc')
            g_losses.append(1e0*G_loss_4)
            d_losses.append(1e0*D_loss_4)

        self.g_loss = tf.reduce_mean(g_losses, name='self.g_loss')
        self.d_loss = tf.reduce_mean(d_losses, name='self.d_loss')
        self.collect_variables() # Overload function to WGAN, see above for more details

        add_moving_summary(self.d_loss, self.g_loss)
        ######################################
        # Visualization
        pz  = tf.zeros_like(pi)
        viz = tf.concat([tf.concat([pi,  pz, pz, pnl, pl, pn], axis=2), # ppc, true_noise, wfly_clean, wfly_corrupted
                         tf.concat([pni, c2, n2, pnl, c1, n1], axis=2), # ppc_denoised, noise_ppc, wfly_denoise, noise_fly
                         ], axis=1)
        viz = 255*(viz)
        viz = tf.cast(tf.clip_by_value(viz, 0, 255), tf.uint8, name='viz')
        tf.summary.image('colorized', viz, max_outputs=50)


    def optimizer(self):
        lr = symbolic_functions.get_scalar_var('learning_rate', 2e-4, summary=True)
        return tf.train.AdamOptimizer(lr, beta1=0.5, epsilon=1e-3)

###############################################################################
class VisualizeRunner(Callback):
    def __init__(self, input, tower_name='InferenceTower', device=0):
        self.dset = input 
        self._tower_name = tower_name
        self._device = device

    def _setup_graph(self):
        self.pred = self.trainer.get_predictor(
            ['image', 'label', 'noise'], ['viz'])

    def _before_train(self):
        pass

    def _trigger(self):
        for lst in self.dset.get_data():
            viz_test = self.pred(lst)
            viz_test = np.squeeze(np.array(viz_test))

            #print viz_test.shape

            self.trainer.monitors.put_image('viz_test', viz_test)


###############################################################################
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu',        default='0', help='comma seperated list of GPU(s) to use.')
    parser.add_argument('--data',  default='data/wfly1/db_train/', required=True, 
                                    help='Data directory, contain trainA/trainB/validA/validB')
    parser.add_argument('--load',   help='Load the model path')
    parser.add_argument('--sample', help='Run the deployment on an instance',
                                    action='store_true')

    args = parser.parse_args()
    # python Exp_FusionNet2D_-VectorField.py --gpu='0' --data='arranged/'

    
    train_ds = get_data(args.data, 
                        isTrain=True, 
                        isValid=False, 
                        isTest=False)
    valid_ds = get_data(args.data.replace('train', 'valid'),
                        isTrain=False, 
                        isValid=True, 
                        isTest=False)


    train_ds  = PrefetchDataZMQ(train_ds, 4)
    train_ds  = PrintData(train_ds)
    # train_ds  = QueueInput(train_ds)
    model     = Model()

    os.environ['PYTHONWARNINGS'] = 'ignore'

    # Set the GPU
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    # Running train or deploy
    if args.sample:
        # TODO
        print("Deploy the data")
        prefix="result/SharedGenerator_Phase1/"
        if os.path.exists(prefix):
            shutil.rmtree(prefix, ignore_errors=True)
        os.makedirs(prefix) 
        sample(args.data, args.load, model, prefix = prefix)
        # pass
    else:
        # Set up configuration
        # Set the logger directory
        logger.auto_set_dir()

        session_init = SaverRestore(args.load) if args.load else None 


        GANTrainer(StagingInput(QueueInput(train_ds)), model).train_with_defaults(
            callbacks       =   [
                PeriodicTrigger(ModelSaver(), every_k_epochs=50),
                PeriodicTrigger(VisualizeRunner(valid_ds), every_k_epochs=5),
                PeriodicTrigger(InferenceRunner(valid_ds, [ScalarStats('loss_mae/mae_wfly1')]), every_k_epochs=1),
                # ScheduledHyperParamSetter('learning_rate', [(0, 1e-6), (300, 1e-6)], interp='linear')
                ScheduledHyperParamSetter('learning_rate', [(0, 2e-4), (100, 1e-4), (200, 1e-5), (300, 1e-6)], interp='linear'),
                # ScheduledHyperParamSetter('learning_rate', [(30, 6e-6), (45, 1e-6), (60, 8e-7)]),
                # HumanHyperParamSetter('learning_rate'),
                ClipCallback(),
                ],
                max_epoch       =   500, 
                session_init    =   session_init,
                steps_per_epoch =   EPOCH_SIZE,
        )
