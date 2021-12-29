from ElectraDataModule import *
from ElectraBinaryClassification import *

if __name__ == "__main__" :
    model = ElectraClassification(learning_rate=0.0001)

    dm = ElectraClassificationDataModule(batch_size=8, train_path='./train.tsv', valid_path='./dev.tsv',
                                    max_length=256, sep='\t', doc_col='comments', label_col='contain_gender_bias', num_workers=1,
                                    labels_dict={False : 0, True : 1})
    
    checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor='val_f1',
                                                    dirpath='../sample_electra_binary_hate_chpt',
                                                    filename='KoELECTRA/{epoch:02d}-{val_f1:.3f}',
                                                    verbose=True,
                                                    save_last=True,
                                                    mode='max',
                                                    save_top_k=-1,
                                                    )
    
    tb_logger = pl_loggers.TensorBoardLogger(os.path.join('../sample_electra_binary_hate_chpt', 'tb_logs'))

    lr_logger = pl.callbacks.LearningRateMonitor()

    trainer = pl.Trainer(
        default_root_dir='../sample_electra_binary_hate_chpt/checkpoints',
        logger = tb_logger,
        callbacks = [checkpoint_callback, lr_logger],
        max_epochs=10,
        gpus=1
    )

    trainer.fit(model, dm)
