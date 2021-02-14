from keras.callbacks import TensorBoard,LambdaCallback

class LearningRateLoggingCallback(tf.keras.callbacks.Callback):

    def on_epoch_end(self, epoch):
        lr = self.model.optimizer._decayed_lr('float32').numpy()
        tf.summary.scalar('learning rate', data=lr, step=epoch)



train_lr_callback = LambdaCallback( on_epoch_begin= lambda epoch,logs: print("LearningRate of %e" % (K.eval(resnet20.optimizer._decayed_lr('float32').numpy())) ))

history = resnet20.fit(train_gen,
                                    epochs=200,
                                    callbacks=[reduce_lr,TensorBoard(log_dir='/content/output'),train_lr_callback],
                                    validation_data=(x_val, y_val)
                      #  , verbose=1
                                    )
