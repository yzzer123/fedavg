#!/usr/bin/env python3
import logging

from utils import Properties
from test.test_pickle import obj_test
logging.getLogger().setLevel(Properties.LOGGING_LEVEL)

from test.train_test import main
from test.test_pickle import performace_test, test_model
from models.ResNet import ResNetCIFAR10
from models.BasicModel import LocalEnvironment

if __name__ == "__main__":
    model = ResNetCIFAR10(3000)
    env = LocalEnvironment()
    model.client_init(env)
    model.to(env.device)
    model.local_train(env)
    
    
    # print(Properties.get(Properties.TRAINER_CLUSTER))
    # obj_test()
    # main()
    # performace_test()
    # test_model()
