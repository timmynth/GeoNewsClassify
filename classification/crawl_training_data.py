import orientaldaily_past_spider
import datetime

""" Craw orientaldaily """
# list of dates backward from today
base = datetime.datetime.today()
numdays = 365 * 4
dates = [base - datetime.timedelta(days=x) for x in range(0, numdays)]
dates = [d.strftime('%Y%m%d') for d in dates]
categories = [
    'news',             #要聞港聞
    'china_world',      #兩岸國際
    'finance',          #財經
    'entertainment',    #娛樂
    'lifestyle',        #副刊
    'sport'             #體育
    ]
for d in dates:
    for c in categories:
        baseUrl = [('http://orientaldaily.on.cc/cnt/'+c+'/'+d+'/', 0)]
        rules = {'link_prefix': ['http://orientaldaily.on.cc/cnt/'+c+'/'+d+'/']}
        spider = orientaldaily_past_spider.Spider(baseUrl=baseUrl, rules=rules)
        spider.crawl_and_save(save_prefix=c+'/')
