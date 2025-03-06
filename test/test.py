import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 只显示错误信息，屏蔽 INFO 和 WARNING
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import numpy as np
from unittest.mock import patch

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from keras.layers.core import Dropout
from keras import backend as K

def test_init_with_noise_shape_and_seed():
    try:
        dropout_layer = Dropout(p=0.5, noise_shape=(3, 1), seed=42)
        assert dropout_layer.p == 0.5
        assert dropout_layer.noise_shape == (3, 1)
        assert dropout_layer.seed == 42
        print("Test 1 passed")
    except:
        print("Test 1 failed")


def test_get_noise_shape_method():
    try:
        # 显式指定 noise_shape
        dropout1 = Dropout(p=0.5, noise_shape=(3, 1))
        assert dropout1._get_noise_shape(None) == (3, 1), "Should return specified noise_shape"
        
        # 不指定 noise_shape（应返回 None）
        dropout2 = Dropout(p=0.5)
        assert dropout2._get_noise_shape(None) is None, "Should return None"
        
        print("Test 2 passed")
    except:
        print("Test 2 failed")



def test_call_method_uses_seed():
    try:
        with open("./keras/layers/core.py", "r") as f:
            content = f.readlines()
        
        for line_idx in range(96, 101):
            if "K.in_train_phase" not in content[line_idx]:
                continue
            else:
                assert "self.seed" in content[line_idx], "not propogate seed to dropout backend"
            
        print("Test 3 passed")
    except Exception as e:
        print(f"Test 3 failed: {e}")

    
if __name__ == "__main__":
    test_init_with_noise_shape_and_seed()
    test_get_noise_shape_method()
    test_call_method_uses_seed()
