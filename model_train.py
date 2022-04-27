import xgboost as xgb
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.neural_network import MLPRegressor

from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from wrangler import Wrangler

if __name__ == '__main__':
    # load data
    nndf = Wrangler.load_pickle('data/nndf_train.pkl')
    # get cols for training
    cols = ['x', 'y', 'd_t-1', 'd_t-2', 'd_t-3', 'd_light', 'l0', 'l1',
            'l2', 'l3', 'dir_0', 'dir_1', 'dir_2'] + ['d_zone_' + str(i) for i in range(6)]
    # init x and y
    x_train = nndf[cols].to_numpy()
    y_train = nndf['target'].to_numpy()

    # do train val test split
    # x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2)

    # create and fit xgb regressor
    # reg = xgb.XGBModel()
    # reg.fit(x_train, y_train, eval_set=[(x_val, y_val)], eval_metric='rmse', verbose=True)
    # reg = LGBMRegressor()
    # reg = CatBoostRegressor()
    reg = MLPRegressor()
    reg.fit(x_train, y_train)

    # save model
    Wrangler.dump_pickle(reg, 'models/cgb.pkl')

    # test model
    nndf = Wrangler.load_pickle('data/nndf_test.pkl')
    # get cols for training
    cols = ['x', 'y', 'd_t-1', 'd_t-2', 'd_t-3', 'd_light', 'l0', 'l1',
            'l2', 'l3', 'dir_0', 'dir_1', 'dir_2'] + ['d_zone_' + str(i) for i in range(6)]
    # init x and y
    x_test = nndf[cols].to_numpy()
    y_test = nndf['target'].to_numpy()

    # test r2_score on test set
    pred = reg.predict(x_test)
    score = r2_score(y_test, pred)
    print(score)
