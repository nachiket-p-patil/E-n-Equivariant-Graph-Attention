import numpy as np
import torch
# import torch.nn as nn

from sklearn.metrics import mean_squared_error,mean_absolute_error
# import xgboost as xgb
# import os
# from data_deal import decrease_learning_rate
from torch.autograd import Variable

def xgb_regression(X_train, X_val, X_test,label, train_split, val_split, test_split, args):
    y_train, y_val, y_test = label[:train_split], label[train_split+1:val_split], label[val_split+1:]
    from xgboost.sklearn import XGBRegressor
    model = XGBRegressor(
        learn_rate=0.1,
        max_depth=4,#4
        min_child_weight=10,
        gamma=1,#1
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.8,
        objective='reg:linear',
        n_estimators=2000,
        tree_method = 'gpu_hist',
        n_gpus = -1
    )
    model.fit(X_train, y_train,eval_set=[(X_val, y_val)], eval_metric='rmse',
                early_stopping_rounds=300)
    y_pred = model.predict(X_test)
    y_test = y_test.astype('float')
    MSE = mean_squared_error(y_test,y_pred)
    RMSE = MSE ** 0.5
    mae = mean_absolute_error(y_test, y_pred)
    return RMSE

def get_feature(model,data, args):
    model.eval()
    i = 0
    with torch.no_grad():
        for k in data:
            feature, A, y = k
            padding = torch.zeros(len(feature), 50, args.hidden - 58)
            init_input = torch.cat((feature, padding), 2)
            init_input, A, feature, y = Variable(init_input.cuda()), Variable(A.cuda()), Variable(feature.cuda()), Variable(y.cuda())
            output, feature = model(init_input, feature, A)
            if i ==0:
                features = feature
            else:
                features = torch.cat((features,feature))
            i = i+1
    return features

def get_xgb_scores(model, args, label, train_dataset, val_dataset, test_dataset, xgb_scores):
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    xgb_train_feature = get_feature(model, train_loader, args)
    xgb_val_feature = get_feature(model, val_loader, args)
    xgb_test_feature = get_feature(model, test_loader, args)
    xgb_train_feature = xgb_train_feature.cpu().numpy()
    xgb_val_feature = xgb_val_feature.cpu().numpy()
    xgb_test_feature = xgb_test_feature.cpu().numpy()
    xgb_RMSE = xgb_regression(xgb_train_feature, xgb_val_feature, xgb_test_feature, label,100000, 18000, 13000, args)
    xgb_scores.append([xgb_RMSE])

    return xgb_scores