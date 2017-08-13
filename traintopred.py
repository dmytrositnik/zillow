def train_to_pred(train, predictions):
    # Remove duplicates from train, takes the latest value.
    train_no_dup = train.drop_duplicates(subset='parcelid', keep='last', inplace=False)

    prediction_rows = predictions.loc[predictions['ParcelId'].isin(train_no_dup['parcelid'])]

    train_no_dup_reindexed = train_no_dup.set_index('parcelid')

    count = 0
    for index, row in prediction_rows.iterrows():
        count += 1
        if count % 1000 == 0:
            print "Index = {}; Count = {}; Logerror = {};".format(
                index,
                count,
                train_no_dup_reindexed.get_value(row['ParcelId'], 'logerror'))

        prediction_rows.set_value(
            index,
            '201610',
            round(train_no_dup_reindexed.get_value(row['ParcelId'], 'logerror'), 4))

    predictions.set_value(
        prediction_rows.index,
        ['201610', '201611', '201612', '201710', '201711', '201712'],
        prediction_rows['201610'])
