###############################################################################
# Copyright (c) 2018-2022 Lawrence Livermore National Security, LLC.
# Produced at the Lawrence Livermore National Laboratory
#
# Written by J. Brodsky, J. Chavez, S. Czyz, G. Kosinovsky, V. Mozin, S. Sangiorgio.
# RASE-support@llnl.gov.
#
# LLNL-CODE-841943, LLNL-CODE-829509
#
# All rights reserved.
#
# This file is part of RASE.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
# of the Software, and to permit persons to whom the Software is furnished to do
# so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED,INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
###############################################################################

"""
This module contains functions copied and slightly modified from LLNL RASE at:
https://github.com/LLNL/RASE/tree/public
"""

from lxml import etree as ET
import numpy as np

# def uncompressCountedZeroes(chanData,counts):
#     if (chanData.attrib.get('Compression') == 'CountedZeroes') or (
#             chanData.attrib.get('compressionCode') == 'CountedZeroes'):
#         uncompressedCounts = []
#         countsIter = iter(counts)
#         for count in countsIter:
#             if count == float(0):
#                 uncompressedCounts.extend([0] * int(next(countsIter)))
#             else:
#                 uncompressedCounts.append(count)
#         counts = ','.join(map(str, uncompressedCounts))
#     else:
#         counts = ','.join(map(str, counts))
#     return counts

def uncompressCountedZeroes(counts):
    uncompressedCounts = []
    countsIter = iter(counts)
    for count in countsIter:
        if count == float(0):
            uncompressedCounts.extend([0] * int(next(countsIter)))
        else:
            uncompressedCounts.append(count)
    return uncompressedCounts


def strip_namespaces(tree:ET.ElementTree):
    # SOURCE: LLNL RASE
    # (put liscence here or whatever)
    tree.getroot()
    query = "descendant-or-self::*[namespace-uri()!='']"
    # for each element returned by the above xpath query...
    for element in tree.xpath(query):
        # replace element name with its local name
        element.tag = ET.QName(element).localname
    tree.getroot().attrib.clear()
    ET.cleanup_namespaces(tree)
    return tree

def rebin(counts, oldEnergies, newEcal):
    """
    Rebins a list of counts to a new energy calibration.

    :param counts:      numpy array of counts indexed by channel
    :param oldEnergies: numpy array of energies indexed by channel
    :param newEcal:     list of new energy polynomial coefficents to rebin to: [E3 E2 E1 E0]
    :return:            numpy array of rebinned counts
    """
    newEnergies = np.polyval(newEcal, np.arange(len(counts)+1))
    newCounts   = np.zeros(len(counts))

    # move old energies index to first value greater than the first value in newEnergies
    oe = 0 # will always lead ne in energy boundary value
    while oe < (len(oldEnergies)-1) and oldEnergies[oe] <= newEnergies[0]: oe += 1
    ne0 = 0
    while ne0 < (len(newEnergies)-1) and (newEnergies[ne0] <= oldEnergies[oe]): 
        # display('l', newEnergies[ne0], oldEnergies[oe])
        ne0 += 1
    # loop through and distribute old counts into new bins
    for ne in range(ne0, len(newCounts)):
        if oe == len(oldEnergies): break  # have already distributed all old counts

        # if no old energy boundaries within this new bin, new bin is fraction of old bin
        if oldEnergies[oe] > newEnergies[ne + 1]:
            newCounts[ne] = counts[oe - 1] * (newEnergies[ne + 1] - newEnergies[ne]) \
                            / (oldEnergies[oe] - oldEnergies[oe - 1])

        # else there are old energy boundaries in this new bin: add each portion of old bins
        else:
            # Step 1: add first partial(or full) old bin
            # TODO: This will crash if (oldEnergies[oe] - oldEnergies[oe-1]) < 0; might be necessary to handle this?
            newCounts[ne] = counts[oe-1] * (oldEnergies[oe] - newEnergies[ne]) \
                            / (oldEnergies[oe] - oldEnergies[oe-1])
            oe += 1

            # Step 2: add middle full old bins
            while oe < len(oldEnergies) and oldEnergies[oe] <= newEnergies[ne+1]:
                newCounts[ne] += counts[oe-1]
                oe += 1
            if oe == len(oldEnergies): break

            # Step 3: add last partial old bin
            newCounts[ne] += counts[oe-1] * (newEnergies[ne+1] - oldEnergies[oe-1]) \
                             / (oldEnergies[oe] - oldEnergies[oe-1])
    return newCounts