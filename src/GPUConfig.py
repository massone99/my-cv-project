import tensorflow as tf


class GPUConfig:
    @staticmethod
    def setup():
        if gpus := tf.config.experimental.list_physical_devices("GPU"):
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                print(e)