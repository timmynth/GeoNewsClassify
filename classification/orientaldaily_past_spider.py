#!/usr/bin/env python
# -*- coding: utf-8 -*-
from html.parser import HTMLParser
from urllib.request import urlopen
from urllib import parse
from bs4 import BeautifulSoup

# We are going to create a class called LinkParser that inherits some
# methods from HTMLParser which is why it is passed into the definition
class LinkParser(HTMLParser): # will be used by apple spider

    # This is a function that HTMLParser normally has
    # but we are adding some functionality to it
    def handle_starttag(self, tag, attrs):
        # We are looking for the begining of a link. Links normally look
        # like <a href="www.someurl.com"></a>
        if tag == 'a':
            for (key, value) in attrs:
                if key == 'href':
                    # We are grabbing the new URL. We are also adding the
                    # base URL to it. For example:
                    # www.netinstructions.com is the base and
                    # somepage.html is the new URL (a relative URL)
                    #
                    # We combine a relative URL with the base URL to create
                    # an absolute URL like:
                    # www.netinstructions.com/somepage.html
                    newUrl = parse.urljoin(self.baseUrl, value)
                    # And add it to our colection of links:
                    if self.rules is not None and self.rules.get('link_prefix') is not None:
                        found = False
                        for rule in self.rules.get('link_prefix'):
                            found = found or newUrl.startswith( parse.urljoin(self.baseUrl, rule ) )
                        if not found:
                            break
                    self.links = self.links + [newUrl]

    # This is a new function that we are creating to get content and links
    # that our spider() function will call
    def get_Content_Links(self, url, rules=None):
        """ Return html string, links """
        self.links = []
        self.rules = rules
        # Remember the base URL which will be important when creating
        # absolute URLs
        self.baseUrl = url
        # Use the urlopen function from the standard Python 3 library
        response = urlopen(url)
        # Make sure that we are looking at HTML and not other things that
        # are floating around on the internet (such as
        # JavaScript files, CSS, or .PDFs for example)
        if response.getheader('Content-Type')=='text/html':
            htmlBytes = response.read()
            # Note that feed() handles Strings well, but not bytes
            # (A change from Python 2.x to Python 3.x)
            htmlString = htmlBytes.decode("utf-8")
            self.feed(htmlString)
            return htmlString, self.links
        else:
            return "",[]

class Spider:

    def __init__(self, baseUrl=None, rules=None, callback=None):
        self.baseUrl = baseUrl or [('http://orientaldaily.on.cc/cnt/finance/20160717/', 0)]
        self.rules = rules or {'link_prefix': ['http://orientaldaily.on.cc/cnt/finance/20160717/']}

    def extract_content_orientaldaily(self, html, url):
        """ Extract oriental daily 1 header, 2 contect """
        soup = BeautifulSoup(html, 'html.parser')
        # print (soup.prettify())
        contents = soup.find_all(['p','h3'])
        content_string = ''
        for c in contents:
            content_string += c.getText() + '\n'
        return content_string

    def crawl_and_save(self, maxLevel=1, save_prefix='finance/'):
        """ Craw the page with maxLevel """
        pagesToVisit = self.baseUrl
        levelVisited = 0
        while pagesToVisit != []:
            url, levelVisited = pagesToVisit[0]
            if levelVisited > maxLevel:
                break
            pagesToVisit = pagesToVisit[1:]
            parser = LinkParser()
            # get html, links
            html, links = parser.get_Content_Links(url, self.rules)
            # get content string from html
            content_string = self.extract_content_orientaldaily(html, url)
            print ('Crawled : ', url)
            # write the string
            if levelVisited > 0 and content_string != '':
                fwrite = open(save_prefix+'/'+url.replace('/', '-')+'.txt', 'w')
                fwrite.write(url+'\n')
                fwrite.write(content_string)
                fwrite.close()
            # Add the pages that we visited to the end of our collection of pages to visit:
            links = [(link, levelVisited+1) for link in links ]
            pagesToVisit = pagesToVisit + links
