import utils
import numpy
import pandas
import scipy.stats as stats
import sys
import csv
import pathos.multiprocessing as multiprocessing
from pathos.multiprocessing import ProcessingPool
from collections import namedtuple
from itertools import chain

#
# http://www.racketracer.com/2016/07/06/pandas-in-parallel/
# https://stackoverflow.com/questions/26059764/python-multiprocessing-with-pathos
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.normaltest.html#scipy-stats-normaltest
#

Sample = namedtuple('Sample', 'description, items, normality')

def contingency_row(pool, index, samples):
    x = samples[index].items
    result = list((1,))
    # H0: tHE 2 independent samples have identical average (expected) values
    before = list(map(lambda y: stats.ttest_ind(x, samples[y].items, equal_var = False)[1], range(0, index)))
    after = list(map(lambda y: stats.ttest_ind(x, samples[y].items, equal_var = False)[1], range(index+1, len(samples))))
    
    result.insert(0, before)
    result.append(after)    
    row = utils.flat(result)
    row.insert(0, samples[index].description)    
    return row

if __name__ == "__main__":
    inventory_file = sys.argv[1]
    crosstab_file = sys.argv[2]
    operator_id = int(sys.argv[3])
    pool = multiprocessing.Pool(multiprocessing.cpu_count())
    alpha = 0.05
    min_sample_length = 7 #In order to perform a valid normality test, we need samples larger than min_sample_length
#
#   Loads inventory and sales csv files.
#    
    inventory_data = pandas.read_csv(inventory_file, parse_dates=[0], infer_datetime_format = True)
    
#
#   Selects all records whose belonging to the specified operator.
#
    filtered_inventory_data = inventory_data[inventory_data.operator_id == operator_id][['product_description', 'product_inventory_quantity']]
 
#
#   Creates relevant products dataset (ie. products which ocurr more than min_sample_length.)
# 
    
    relevant_products = filtered_inventory_data \
        .groupby(('product_description',)) \
        .filter(lambda r: r['product_inventory_quantity'].count() > min_sample_length) \
        .drop('product_inventory_quantity', axis=1)

#
#   Extracts inventory quantities for relevant products
#
        
    relevant_inventory_data = filtered_inventory_data.merge( relevant_products, on = ('product_description',))    
    relevant_inventory_quantities = tuple(
        map(
            lambda o: (
                o[0],
                tuple(map(lambda r: relevant_inventory_data.iloc[o[1][r]]['product_inventory_quantity'], range(0, len(o[1]))))
            ),
            relevant_inventory_data.groupby('product_description').groups.items()
        )
    )

#
#   Create samples array (1 sample per product) and perform normality test on each sample
#
    samples = pool.map(lambda x: Sample(x[0], x[1], stats.normaltest(x[1])), relevant_inventory_quantities) 
    normal_samples = tuple(filter(lambda s: s.normality.pvalue > alpha , samples)) #  HO: s comes from a normal distribution
    f, p = stats.f_oneway(*pool.map(lambda s: s.items, normal_samples))
    
    
    if p < alpha: #  H0: All population means are equal, in this case we are concerneed only with instances where H1 holds.
        table = list(map(lambda index: contingency_row(pool, index, normal_samples) , range(0, len(normal_samples))))
        labels = list(map(lambda s: s.description, normal_samples))
        labels.insert(0, 'Product')
        result = pandas.DataFrame(table)
        result.columns = labels
        result.set_index('Product')
        result.to_csv(crosstab_file, quoting = csv.QUOTE_NONNUMERIC)