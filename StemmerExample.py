#!/usr/bin/env python
# encoding: utf-8
"""
abc.py

Created by Andrew John Tulloch on 2010-05-20.
Copyright (c) 2010 Andrew Tulloch. All rights reserved.
"""

import sys
import os
import Stemmer

def main():
	stemmer = Stemmer.Stemmer("english")
	print stemmer.stemWord("cardsing")
	


if __name__ == '__main__':
	main()

