from whoosh.qparser import QueryParser
from whoosh import scoring
from whoosh.index import open_dir
import sys
 
ix = open_dir("indexdir")
 
# query_str is query string
query_str = sys.argv[1]
# Top 'n' documents as result
topN = int(sys.argv[2])
 
with ix.searcher(weighting=scoring.Frequency) as searcher:
    query = QueryParser("content", ix.schema).parse(query_str)
    results = searcher.search(query,limit=topN)
    res_len = len(results)
    real_top = min(topN,res_len)
    print('Топ {:d}/{:d}:\n'.format(real_top,res_len))
    for i in range(real_top):
        # print(results[i]['title'], str(results[i].score), results[i]['textdata'])
        print('{:>9} {:4.1f}'.format(results[i]['title'], results[i].score))