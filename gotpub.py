#!/usr/bin/env python
# coding: utf-8

"""
Modified script to fetch melting point data from PubChem for compounds in CSV file
Rate limited to 2 requests per second
"""

import re
import sys
import traceback
import xml.etree.ElementTree as ET
from typing import Optional
import time
import signal

import pandas as pd
import pubchempy as pcp
import requests


debug = False
results = []  # Global to access in signal handler


def save_results(output_file):
    """Save results to CSV file"""
    if results:
        results_df = pd.DataFrame(results)
        results_df.to_csv(output_file, index=False)
        print(f"\n{'='*60}")
        print(f"Results saved to: {output_file}")
        print(f"Total compounds saved: {len(results)}")
        print(f"{'='*60}")
    else:
        print("\nNo results to save!")


def signal_handler(sig, frame):
    """Handle keyboard interrupt (Ctrl+C)"""
    print('\n\n⚠ Keyboard interrupt detected! Saving progress...')
    output_file = 'main-data/melting_point_results.csv'
    save_results(output_file)
    print('\nExiting...')
    sys.exit(0)


def get_melting_point_from_pubchem(cid) -> Optional[dict]:
    """
    Look up melting point for a given PubChem CID
    
    Args:
        cid: PubChem Compound ID
        
    Returns:
        Dictionary with compound info and melting point, or None if not found
    """
    global debug
    
    lookup_source = 'Pubchem'

    try:
        headers = {
            'user-agent': 'Mozilla/5.0 (X11; CentOS; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/73.0.3683.75 Safari/537.36'
        }

        # Get compound properties
        lookup_result = pcp.get_properties(['inchi', 'inchikey',
                                    'canonical_smiles', 'isomeric_smiles',
                                    'iupac_name'],
                            cid)
        
        if not lookup_result:
            raise RuntimeError(f'Compound with CID {cid} not found in Pubchem.')
        
        # Get synonyms to extract CAS number
        synonyms = pcp.get_synonyms(cid)[0]['Synonym'] or []
        
        returned_cas = ''
        for synonym in synonyms:
            cas_nr = re.search(r'^\d{2,7}-\d{2}-\d$', synonym)
            if cas_nr:
                cas_nr = cas_nr.group()
                returned_cas = cas_nr
                break

        # Get melting point data from PubChem
        melting_point_lookup_result_xml = f'https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data/compound/{cid}/XML?heading=Melting+Point'

        r = requests.get(melting_point_lookup_result_xml, headers=headers, timeout=15)
        
        if r.status_code == 200 and len(r.history) == 0:
            tree = ET.fromstring(r.text)
            info_node = tree.find('.//*{http://pubchem.ncbi.nlm.nih.gov/pug_view}Information')

            if info_node is not None:
                original_source_elem = info_node.find('{http://pubchem.ncbi.nlm.nih.gov/pug_view}Value')

                melting_point_elem1 = info_node.find('.//*{http://pubchem.ncbi.nlm.nih.gov/pug_view}String')
                melting_point_elem2 = info_node.find('.//*{http://pubchem.ncbi.nlm.nih.gov/pug_view}Number')
                original_source = original_source_elem.text if original_source_elem is not None else ''
                melting_point_raw = melting_point_elem1.text if melting_point_elem1 is not None else ''

                # Extract temperature value using regex (e.g., "34.5 °C", "-10 °F", "100.2 °C")
                melting_point_result = ''
                if melting_point_raw:
                    # Pattern matches: optional minus, digits with optional decimal, space, degree symbol, C or F
                    temp_match = re.search(r'-?\d+(?:\.\d+)?\s*°[CF]', melting_point_raw)
                    if temp_match:
                        melting_point_result = temp_match.group()
                    else:
                        # If no match, keep the original value
                        melting_point_result = melting_point_raw
                else:
                    # Try to extract from <Number> and <Unit> structure
                    # e.g., <Value><Number>204</Number><Unit>°C</Unit></Value>
                    number_elem = info_node.find('.//*{http://pubchem.ncbi.nlm.nih.gov/pug_view}Number')
                    unit_elem = info_node.find('.//*{http://pubchem.ncbi.nlm.nih.gov/pug_view}Unit')
                    
                    if number_elem is not None and unit_elem is not None:
                        number_text = number_elem.text if number_elem.text else ''
                        unit_text = unit_elem.text if unit_elem.text else ''
                        if number_text and unit_text:
                            melting_point_result = f"{number_text} {unit_text}"
                
                core_result = {
                    'source': lookup_source,
                    'Pubchem_CID': str(cid),
                    'Melting_Point': melting_point_result,
                    'reference': original_source,
                    'Substance_CASRN': returned_cas,
                }
                extra_info = lookup_result[0]
                extra_info.pop('CID', None)

                result = {**core_result, **extra_info}
                
                # Rename keys
                s = pd.Series(result)
                s = s.rename({
                    'CanonicalSMILES': 'Canonical_SMILES',
                    'IsomericSMILES': 'Isomeric_SMILES',
                    'IUPACName': 'IUPAC_Name'
                })
                result = s.to_dict()            
                return result
            else:
                # No melting point data found, but return basic compound info
                core_result = {
                    'source': lookup_source,
                    'Pubchem_CID': str(cid),
                    'Melting_Point': None,
                    'reference': None,
                    'Substance_CASRN': returned_cas,
                }
                extra_info = lookup_result[0]
                extra_info.pop('CID', None)

                result = {**core_result, **extra_info}
                s = pd.Series(result)
                s = s.rename({
                    'CanonicalSMILES': 'Canonical_SMILES',
                    'IsomericSMILES': 'Isomeric_SMILES',
                    'IUPACName': 'IUPAC_Name'
                })
                result = s.to_dict()
                return result
        else:
            # Request failed, but return basic compound info
            core_result = {
                'source': lookup_source,
                'Pubchem_CID': str(cid),
                'Melting_Point': None,
                'reference': None,
                'Substance_CASRN': returned_cas,
            }
            extra_info = lookup_result[0]
            extra_info.pop('CID', None)

            result = {**core_result, **extra_info}
            s = pd.Series(result)
            s = s.rename({
                'CanonicalSMILES': 'Canonical_SMILES',
                'IsomericSMILES': 'Isomeric_SMILES',
                'IUPACName': 'IUPAC_Name'
            })
            result = s.to_dict()
            return result

    except Exception as error:
        if debug:
            traceback_str = ''.join(traceback.format_exception(etype=type(error), value=error, tb=error.__traceback__))
            print(f"Error processing CID {cid}: {traceback_str}")
        else:
            print(f"Error processing CID {cid}: {error}")

        return None


def main():
    """
    Main function to read CSV, fetch melting points, and save results
    """
    global results
    
    # Register signal handler for Ctrl+C
    signal.signal(signal.SIGINT, signal_handler)
    
    # Read the input CSV file
    input_file = 'main-data/PubChem_compound_cache_aC_PHLvv3lPpedxgXhiVR5ruKI72DngrAg5jZxkfcWYZBk0.csv'
    output_file = 'main-data/melting_point_results.csv'
    
    print(f"Reading compounds from {input_file}...")
    df = pd.read_csv(input_file)
    
    # Extract CIDs from the Compound_CID column
    cids = df['Compound_CID'].tolist()
    total_compounds = len(cids)
    
    # Load existing results and determine where to resume
    last_stop = pd.read_csv(output_file)
    last_idx = last_stop.shape[0] if not last_stop.empty else 0
    
    # Initialize results with existing data
    results = last_stop.to_dict('records') if not last_stop.empty else []
    
    if last_idx > 0:
        print(f"Resuming from compound {last_idx + 1}/{total_compounds}")
        print(f"Already processed: {last_idx} compounds\n")
    
    print(f"Found {total_compounds} compounds to process")
    print("Rate limit: 2 requests per second")
    print("Press Ctrl+C to stop and save progress\n")
    print("Starting data collection...\n")
    
    start_time = time.time()

    try:
        # Start from last_idx instead of 0
        for idx, cid in enumerate(cids[last_idx:], last_idx + 1):
            print(f"Processing {idx}/{total_compounds}: CID {cid}")
            
            result = get_melting_point_from_pubchem(cid)
            
            if result:
                results.append(result)
                print(f"  ✓ Success - Melting Point: {result.get('Melting_Point', 'N/A')}")
            else:
                print(f"  ✗ Failed to retrieve data")
            
            # Rate limiting: 2 requests per second = 0.5 seconds between requests
            time.sleep(0.35)
    
    except KeyboardInterrupt:
        # This will be caught by signal handler
        pass
    
    # Save results to CSV (normal completion)
    save_results(output_file)
    print(f"Total time: {time.time() - start_time:.2f} seconds")


if __name__ == "__main__":
    if '--debug' in sys.argv or '-d' in sys.argv:
        debug = True
        print("Debug mode enabled\n")
    
    main()
