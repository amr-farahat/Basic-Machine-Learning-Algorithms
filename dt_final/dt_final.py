from __future__ import division
from numpy import genfromtxt
from math import log
import xml.etree.ElementTree as ET
from xml.dom import minidom
import sys

# asking the user for the input data file name and the desired name for the opuput xml file. 
in_file = raw_input("Enter the input data File Name: ")
out_file = raw_input("Enter the output xml File Name: ")


classes = ['unacc', 'acc', 'good', 'vgood']

attributes = {'buying': ['vhigh', 'high', 'med', 'low'],
              'maint': ['vhigh', 'high', 'med', 'low'],
              'doors': ['2', '3', '4', '5more'],
              'persons': ['2', '4', 'more'],
              'lug_boot': ['small', 'med', 'big'],
              'safety': ['low', 'med', 'high']}
attributes_list = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety']

# loading the data and formatting it as a tuple of attributes and class.
try:
    cardata = genfromtxt(in_file, dtype=None, delimiter=',')
except IOError:
    print "File", in_file ,"not found"
    sys.exit(0)
formatted_data = [(i[0:6], i[6]) for i in cardata]


def entropy(S):
    ''' entropy is a function that takes a set/subset of the training
 	examoles S and returns a list contains the total count of the instances in the set, the entropy and the counts of different classes in the set '''
    total_count = len(S)
    p = []
    counts = []
    for c in classes:
        count = len([i for i in S if i[1] == c])
        counts.append(count)
        p.append(count / total_count)
    return [total_count, sum([-1 * i * log(i, len(classes)) for i in p if i]), counts]


def partition(S, A):
    ''' partition is a function that takes a set/subset of the training 
	examples and attribute A and returns multiple sets partitioned 
	by the values of this attribute A'''
    keys = attributes[A]
    sets = {key: [] for key in keys}
    for s in S:
        key = s[0][attributes_list.index(A)]
        sets[key].append(s)
    return sets


def gain(S, A):
    ''' gain is a function that takes a set/subset of the training 
	examples and attribute A and returns the information
	gain from using this attribute as a node'''
    total_count, ent, counts = entropy(S)
    sets = partition(S, A)
    entropies = []
    total_counts = []
    for s in sets:
        temp1, temp2, counts = entropy(sets[s])
        total_counts.append(temp1)
        entropies.append(temp2)
    return ent - sum([(total_counts[i] / total_count) * entropies[i]
                      for i in range(len(total_counts))])


def best_attribute(S, attributes_list):
    ''' best_attribute is a function that takes a set/subset of the training
 	examples and an attribute list and returns the best of 
	these attributes to partition the data upon'''
    gains = []
    for attribute in attributes_list:
        gains.append(gain(S, attribute))
    return attributes_list[gains.index((max(gains)))]


ent = entropy(formatted_data)
c = formatted_data[0][1]
cs = [classes[i] + ':' + str(ent[2][i])
      for i in range(len(classes)) if ent[2][i]]
xml = ET.Element('tree', attrib={'classes': ",".join(cs),
                                 'entropy': str(round(ent[1], 3))})


def id3(S, candidates, xml):
    ''' id3 is a function that takes set/subset of the training
 	examples, candidate attributes, xml root element 
	and returns full xml tree'''
    # check if we reached perfect classification to return a leaf.
    ent = entropy(S)
    if not ent[1]:
        c = S[0][1]
        xml.text = c
        return xml

    # check if we don;t have any attributes left to return a leaf with the
    # majority vote.
    if not candidates:
        ent = entropy(S)
        counts = ent[2]
        majority_count = max(counts)
        majority_class = classes[counts.index(majority_count)]
        xml.text = majority_class
        return xml

    # if not we choose the best atrribute, partition the data over its values
    # and run the function again over the new sets (recursion) wwith less
    # attributes.
    best = best_attribute(S, candidates)
    sets = partition(S, best)
    new_candidates = [a for a in candidates if a != best]
    for value in attributes[best]:
        ent = entropy(sets[value])
        cs = [classes[i] + ':' + str(ent[2][i])
              for i in range(len(classes)) if ent[2][i]]
        tag = ET.SubElement(xml, 'node', attrib={'classes': ",".join(cs),
                                                 'entropy': str(round(ent[1], 3)),
                                                 best: value})
        id3(sets[value], new_candidates, tag)
    return xml

# run the id3 function with the formatted data.
xml = id3(formatted_data, attributes_list, xml)
# convert the xml tree object into a string and dump it into a file.
xml_str = minidom.parseString(ET.tostring(xml)).toprettyxml(indent="   ")
with open(out_file, "w") as f:
    f.write(xml_str)



