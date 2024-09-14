import json
import numpy as np
import os
import pandas as pd

if __name__ == '__main__':
    folder = './experiments'
    dataset = 'REFIT'
    appliance = 'kettle'
    file_name = 'test_result.json'

    length = 16334270

    file_path = os.path.join(folder, dataset, appliance, file_name)
    with open(file_path) as f:
        content = json.load(f)

        real = np.array(content['gt'])
        pred = np.array(content['pred'])
        aggregate = np.array(content['aggregate'])

        df = pd.DataFrame(aggregate, columns=['aggregate'])
        print(df)
        df = df.dropna().copy()
        print(df)
        df = df[df['aggregate'] > 0]
        print(df)

        mae = np.mean(np.abs(pred - real))

        print(aggregate.shape)
        print(pred.shape)

