import random

from database.feverous_db import FeverousDB
from utils.wiki_page import WikiPage
import jsonlines
import json
import unicodedata
from cleantext import clean
import unicodedata
from urllib.parse import unquote
import logging
import numpy as np

DB = None

def clean_title(text):
    text = unquote(text)
    text = clean(text.strip(),fix_unicode=True,               # fix various unicode errors
    to_ascii=False,                  # transliterate to closest ASCII representation
    lower=False,                     # lowercase text
    no_line_breaks=False,           # fully strip line breaks as opposed to only normalizing them
    no_urls=True,                  # replace all URLs with a special token
    no_emails=False,                # replace all email addresses with a special token
    no_phone_numbers=False,         # replace all phone numbers with a special token
    no_numbers=False,               # replace all numbers with a special token
    no_digits=False,                # replace all digits with a special token
    no_currency_symbols=False,      # replace all currency symbols with a special token
    no_punct=False,                 # remove punctuations
    replace_with_url="<URL>",
    replace_with_email="<EMAIL>",
    replace_with_phone_number="<PHONE>",
    replace_with_number="<NUMBER>",
    replace_with_digit="0",
    replace_with_currency_symbol="<CUR>",
    lang="en"                       # set to 'de' for German special handling
    )
    return text


def init_db(wiki_path):
    global DB
    DB = FeverousDB(wiki_path)
    return DB

def get_wikipage_by_id(id):
    page = id.split('_')[0]
    # page = clean_title(page) legacy function used for old train/dev set. Not needed with current data version.
    page = unicodedata.normalize('NFD', page).strip()
    lines = DB.get_doc_json(page)

    if lines == None:
        print('Could not find page in database. Please ensure that the title is formatted correctly. If you using an old version (earlier than 04. June 2021, dowload the train and dev splits again and replace them in the directory accordingly.')
    pa = WikiPage(page, lines)
    return pa

def get_evidence_text_by_id(id, wikipage):
    id_org = id
    id = '_'.join(id.split('_')[1:])
    if id.startswith('cell_') or id.startswith('header_cell_'):
        content = wikipage.get_cell_content(id)
    elif id.startswith('item_'):
        content = wikipage.get_item_content(id)
    elif '_caption' in id:
        content = wikipage.get_caption_content(id)
    else:
        if id in wikipage.get_page_items(): #Filters annotations that are not in the most recent Wikidump (due to additionally removed pages)
            content = str(wikipage.get_page_items()[id])
        else:
            print('Evidence text: {} in {} not found.'.format(id, id_org))
            content = ''
    return content

def get_evidence_by_table(evidence):
    evidence_by_table = {}
    for ev in evidence:
        if '_cell_' in ev:
            table = ev.split("_cell_")[1].split('_')[0]
        elif '_caption_' in ev:
            table = ev.split("_caption_")[1].split('_')[0]
        else:
            continue
        if table in evidence_by_table:
            evidence_by_table[table].append(ev)
        else:
            evidence_by_table[table] = [ev]
    return [list(values) for key, values in evidence_by_table.items()]

def get_evidence_by_page(evidence):
    evidence_by_page = {}
    for i, ele in enumerate(evidence):
        page = ele.split("_")[0]
        # page = str(i)
        if page in evidence_by_page:
            evidence_by_page[page].append(ele)
        else:
            evidence_by_page[page] = [ele]
    evis = [list(values) for key, values in evidence_by_page.items()]
    random.shuffle(evis)
    return evis


def calculate_header_type(header_content):
    real_count = 0
    text_count  = 0
    for ele in header_content:
        if ele.replace('.','',1).isdigit():
            real_count+=1
        else:
            text_count+=1
    if real_count >= text_count:
        return 'real'
    else:
        return 'text'

def group_evidence_by_header(table):
    cell_headers = {}
    for ele in table:
        if 'header_cell_' in ele:
            continue #Ignore evidence header cells for now, probably an exception anyways
        else:
            wiki_page = get_wikipage_by_id(ele)
            cell_header_ele = [ele.split('_')[0] + '_' +  el.get_id().replace('hc_', 'header_cell_') for el in wiki_page.get_context('_'.join(ele.split('_')[1:])) if "header_cell_" in el.get_id()]
            for head in cell_header_ele:
                if head in cell_headers:
                    cell_headers[head].append(get_evidence_text_by_id(ele, wiki_page))
                else:
                    cell_headers[head] = [get_evidence_text_by_id(ele, wiki_page)]
    cell_headers_type = {}
    for ele, value in cell_headers.items():
        cell_headers[ele] = set(value)

    for key,item in cell_headers.items():
        cell_headers_type[key] = calculate_header_type(item)

    return cell_headers, cell_headers_type

def find_headers(cell, page):
    table = None
    for ele in page.get_tables():
        if ele.name.split('_')[-1] == cell.split('_')[1]:
            table = ele
            break
    if table is None:
        logging.warning("Table not found in context, {}".format(cell))
        # return None, None

    cell_row = table.all_cells[cell].row_num
    cell_col = table.all_cells[cell].col_num
    headers_row = [cell for i, cell in enumerate(table.rows[cell_row].row) if cell_col > i]
    headers_row.reverse()
    context_row = set([])
    encountered_header = False
    for ele in headers_row:
        if ele.is_header:
            context_row.add(ele)
            encountered_header = True
        elif encountered_header:
            break

    if not context_row:
        try:
            for idx in range(cell_row):
                if table.rows[cell_row].row[idx].content:
                    context_row.add(table.rows[cell_row].row[idx])
                    break
        except IndexError:
            pass
    if not context_row:
        context_row.add(table.rows[cell_row].row[0])

    headers_column = [row.row[cell_col] for row in table.rows if cell_row > row.row_num]
    headers_column.reverse()
    context_column = set([])
    encountered_header = False
    for ele in headers_column:
        if ele.is_header:
            context_column.add(ele)
            encountered_header = True
        elif encountered_header:
            break
    if not context_column:
        try:
            for idx in range(cell_col):
                if table.rows[idx].row[cell_col].content:
                    context_column.add(table.rows[idx].row[cell_col])
                    break
        except IndexError:
            pass
    if not context_column:
        context_column.add(table.rows[0].row[cell_col])

    return table.type, [list(context_row), list(context_column)]

    # if table.type == "general" or table.type == "infobox":
    #     return table.type, [list(context_row), list(context_column)]
    # else:
    #     print(table.type)
    #     assert False

def group_evidence_by_header_2d(table):
    def get_cell_name(ele, header):
        return ele.split('_')[0] + '_' + header.get_id().replace('hc_', 'header_cell_')

    table_type = ''
    cell_headers = {}
    for ele in table:
        wiki_page = get_wikipage_by_id(ele)
        if 'header_cell_' in ele:
            continue #Ignore evidence header cells for now, probably an exception anyways
        elif "_item_" in ele:
            # continue
            context = wiki_page.get_context('_'.join(ele.split('_')[1:]))
            caption_str = ''
            for ct in context:
                if 'title' in ct.name:
                    caption_str += f"Title : {ct.content} , "
                elif 'section' in ct.name:
                    caption_str += f"Section : {ct.content} , "
                else:
                    assert False, ct.name
            caption_str += "Item : " + get_evidence_text_by_id(ele, wiki_page)
            # caption_str = [None, None, caption_str]
            if "item" in cell_headers:
                cell_headers["item"].append(caption_str)
            else:
                cell_headers["item"] = [caption_str]

        elif "_caption_" in ele:
            # continue
            context = wiki_page.get_context('_'.join(ele.split('_')[1:]))
            caption_str = ''
            for ct in context:
                if 'title' in ct.name:
                    caption_str += f"Title : {ct.content} , "
                elif 'section' in ct.name:
                    caption_str += f"Section : {ct.content} , "
                else:
                    assert False, ct.name
            caption_str += "Caption : " + get_evidence_text_by_id(ele, wiki_page)
            # caption_str = [None, None, caption_str]
            cell_headers["caption"] = [caption_str]
        else:
            table_type, headers = find_headers('_'.join(ele.split('_')[1:]), wiki_page)
            # if table_type is None:
            #     assert False, print(ele)
            #     continue
            row_headers, col_headers = headers
            # row_headers = [ele.split('_')[0] + '_' + row_header.get_id().replace('hc_', 'header_cell_') for row_header in row_headers]
            # col_headers = [ele.split('_')[0] + '_' + col_header.get_id().replace('hc_', 'header_cell_') for col_header in col_headers]
            row_header = row_headers[0]
            col_header = col_headers[0]
            cell_header_ele = []
            # if table_type == "infobox":
            #     cell_header_ele.append(get_cell_name(ele, row_header))
            if "header_cell_" in col_header.get_id():
                cell_header_ele.append(get_cell_name(ele, col_header))
            elif "header_cell_" in row_header.get_id():
                cell_header_ele.append(get_cell_name(ele, row_header))
            else:
                cell_header_ele.append(get_cell_name(ele, col_header))

            # cell_header_ele = [ele.split('_')[0] + '_' +  el.get_id().replace('hc_', 'header_cell_') for el in wiki_page.get_context('_'.join(ele.split('_')[1:])) if "header_cell_" in el.get_id()]
            for head in cell_header_ele:
                cell_item = (get_evidence_text_by_id(get_cell_name(ele, row_header), wiki_page)
                             , get_evidence_text_by_id(get_cell_name(ele, col_header), wiki_page)
                             , get_evidence_text_by_id(ele, wiki_page))
                if head in cell_headers:
                    cell_headers[head].append(cell_item)
                else:
                    cell_headers[head] = [cell_item]
    cell_headers_type = {}
    for ele, value in cell_headers.items():
        cell_headers[ele] = set(value)

    # for key,item in cell_headers.items():
    #     cell_headers_type[key] = calculate_header_type(item)

    return cell_headers, cell_headers_type, table_type


def prepare_input_schlichtkrull(annotation, gold):
    sequence = [annotation.claim]
    if gold:
        evidence_by_page = get_evidence_by_page(annotation.flat_evidence)
    else:
        evidence_by_page = get_evidence_by_page(annotation.predicted_evidence)
    for ele in evidence_by_page:
        for evid in ele:
            wiki_page = get_wikipage_by_id(evid)
            if '_sentence_' in evid:
                sequence.append('. '.join([str(context) for context in wiki_page.get_context(evid)[1:]]) + ' ' + get_evidence_text_by_id(evid, wiki_page))
        tables = get_evidence_by_table(ele)

        for table in tables:
            sequence += linearize_cell_evidence(table)

    # print(sequence)
    return ' </s> '.join(sequence)

def get_merged_linearized_table(entry):

    def merge_tables(tabs_from_cells, tabs_from_sents):
        max_col_num = max([len(tfc[1][0]) for tfc in tabs_from_cells]) if tabs_from_cells else 1
        linearized_lines = []
        for wiki_title, table in tabs_from_sents:
            col_num = 1
            padding_str = ' | ' * (max_col_num - col_num)
            linearized_lines.append("[T] " + wiki_title + padding_str)
            linearized_lines.extend([(row.replace(" | ", " ") + padding_str) for row in table])

        for wiki_title, table in tabs_from_cells:
            col_num = len(table[0])
            padding_str = ' | ' * (max_col_num - col_num)
            linearized_lines.append(' | '.join(["[T] " + wiki_title] * col_num) + padding_str)
            linearized_lines.extend([' | '.join(["[H] " + cell['value'].replace(" | ", " ")
                                                 if cell["is_header"] else cell["value"].replace(" | ", " ")
                                                 for cell in row]) + padding_str for row in table])

        linearized_table = '\n'.join(linearized_lines)
        # [row.split(' | ') for row in linearized_table.split("\n")]
        return linearized_table

    cells = [ele for ele in entry["predicted_evidence"] if "_cell_" in ele]
    cells = list(set(cells))
    grouped_cells = {}
    for cell in cells:
        wiki_title = cell.split("_")[0]
        table_id = cell.split("_cell_")[1].split("_")[0]
        table_key = (wiki_title, table_id)
        cell_id = '_'.join(cell.split("_")[1:])
        if table_key in grouped_cells:
            grouped_cells[table_key].append(cell_id)
        else:
            grouped_cells[table_key] = [cell_id]


    tabs_from_cells = []
    for table_key, cell_ids in grouped_cells.items():
        wiki_title, table_id = table_key
        page_json = DB.get_doc_json(wiki_title)
        wiki_page = WikiPage(wiki_title, page_json)
        rtr_table = wiki_page.get_table_from_cell_id(cell_ids[0])
        keep_columns = []
        keep_rows = []

        for i, row in enumerate(rtr_table.rows):
            for j, c in enumerate(row.row):
                if c.name in cell_ids:
                    keep_rows.append(i)
                    keep_columns.append(j)
                    # break

        keep_rows = sorted(list(set(keep_rows)))
        keep_columns = sorted(list(set(keep_columns)))
        trunc_table = np.array(rtr_table.table)[keep_rows][:,keep_columns].tolist()
        tabs_from_cells.append((wiki_title, trunc_table))

    tabs_from_sents = []
    sents = [ele for ele in entry["predicted_evidence"] if "_sentence_" in ele]
    sents = list(set(sents))
    grouped_sents = {}
    for sent in sents:
        wiki_title = sent.split("_")[0]
        sent_id = '_'.join(sent.split("_")[1:])
        if wiki_title in grouped_sents:
            grouped_sents[wiki_title].append(sent_id)
        else:
            grouped_sents[wiki_title] = [sent_id]

    for wiki_title, sent_ids in grouped_sents.items():
        page_json = DB.get_doc_json(wiki_title)
        wiki_page = WikiPage(wiki_title, page_json)
        sents = []
        # sents.append(wiki_title)
        for sent_id in sent_ids:
            if sent_id in wiki_page.page_order:
                content = str(wiki_page.get_page_items()[sent_id])
                sents.append(content)
            else:
                print('Evidence text: {} in {} not found.'.format(sent_id, wiki_page.title.content))
                content = ''
        if sents:
            tabs_from_sents.append((wiki_title, sents))

    linearized_table = merge_tables(tabs_from_cells, tabs_from_sents)
    return linearized_table

def prepare_input_all2tab(annotation, gold):
    if gold is None:
        if hasattr(annotation, "predicted_evidence"):
            entry = {"predicted_evidence": annotation.predicted_evidence}
        else:
            entry = {"predicted_evidence": annotation.evidence[0]}
    else:
        if not gold:
            entry = {"predicted_evidence": annotation.predicted_evidence}
        else:
            entry = {"predicted_evidence": annotation.evidence[0]}
    linearized_table = get_merged_linearized_table(entry)
    return linearized_table

def prepare_input_all2text(annotation, gold):
    sequence = [annotation.claim]
    if gold:
        evidence_by_page = get_evidence_by_page(annotation.flat_evidence)
    else:
        evidence_by_page = get_evidence_by_page(annotation.predicted_evidence)
    for ele in evidence_by_page:
        for evid in ele:
            wiki_page = get_wikipage_by_id(evid)
            if '_sentence_' in evid:
                sequence.append('. '.join(
                    [str(context) for context in wiki_page.get_context(evid)[1:]]) + ' ' + get_evidence_text_by_id(evid,
                                                                                                                   wiki_page))
        tables = get_evidence_by_table(ele)

        for table in tables:
            sequence += linearize_cell_evidence_2d(table)

    # print(sequence)
    return ' </s> '.join(sequence)

def linearize_cell_evidence(table):
    context = []
    caption_id = [ele for ele in table if '_caption_' in ele]
    context.append(table[0].split('_')[0])
    if len(caption_id) > 0:
        wiki_page = get_wikipage_by_id(caption_id[0])
        context.append(get_evidence_text_by_id(caption_id[0], wiki_page))
    cell_headers, cell_headers_type = group_evidence_by_header(table)
    for key, values in cell_headers.items():
        wiki_page = get_wikipage_by_id(key)
        lin = ''
        key_text = get_evidence_text_by_id(key, wiki_page)
        # print(key, key_text, values)
        for i, value in enumerate(values):
            lin += key_text.split('[H] ')[1].strip() + ' is ' + value #+ ' : ' + cell_headers_type[key]
            if i + 1 < len(values):
                lin += ' ; '
            else:
                lin += '.'

        context.append(lin)
    return context

def linearize_cell_evidence_2d(table):
    context = []
    caption_id = [ele for ele in table if '_caption_' in ele]
    wiki_title = table[0].split("_")[0]
    context.append(wiki_title)
    if len(caption_id) > 0:
        wiki_page = get_wikipage_by_id(caption_id[0])
        context.append(get_evidence_text_by_id(caption_id[0], wiki_page))
    cell_headers, cell_headers_type, table_type = group_evidence_by_header_2d(table)
    for key, values in cell_headers.items():
        if key == "caption":
            context.append(list(values)[0])
            continue
        if key == "item":
            lin = ''
            for i, value in enumerate(values):
                lin += value.strip()
                if i + 1 < len(values):
                    lin += ' ; '
            context.append(lin)
            continue

        wiki_page = get_wikipage_by_id(key)
        lin = ''
        key_text = get_evidence_text_by_id(key, wiki_page)
        # print(key, key_text, values)
        for i, value in enumerate(values):
            row_header, col_header, val = value

            if table_type == "infobox":
                lin += col_header.replace("[H] ", '').strip() + " : " + row_header.replace("[H] ",
                                                                                           '').strip() + " of " + wiki_title + " is " + val.strip()
                # lin += row_header.replace("[H] ", '').strip() + " of " + wiki_title + " is " + val.strip()
            elif table_type == "general":
                lin += col_header.replace("[H] ", '').strip() + " for " + row_header.replace("[H] ",
                                                                                             '').strip() + " is " + val.strip()
                lin = lin.replace("for  ", '')
                # lin += col_header.replace("[H] ", '').strip() + " for " + row_header.replace("[H] ", '').strip() + " of " + wiki_title + " is " + val.strip()
                # lin += key_text.split('[H] ')[1].strip() + ' is ' + value #+ ' : ' + cell_headers_type[key]
            else:
                print(table_type)
                # assert False
            if i + 1 < len(values):
                lin += ' ; '
            else:
                lin += '.'
        context.append(lin)
    return context


def prepare_input(annotation, model_name, gold=False):
    if model_name == 'tabert':
        return prepare_tabert_input(annotation, gold)
    elif model_name == 'schlichtkrull':
        return prepare_input_schlichtkrull(annotation, gold)
    elif model_name == 'all2text':
        return prepare_input_all2text(annotation, gold)
    elif model_name == "all2tab":
        return prepare_input_all2tab(annotation, gold)


def remove_bracket_w_nonascii(s):
    '''
    remove bracketed sub-strings with more than 2 or more than 10% non ascii charactors
    :param s:
    :return: cleaned s
    '''
    p = 0
    lrb = s.find("(")
    while lrb != -1:
        rrb = s.find(")", lrb)
        if rrb == -1:
            break
        ori_r = s[lrb:rrb+1]
        new_r = []

        for r in ori_r[1:-1].split(';'):
            innormal_chars_num = ascii(r).count("\\u") + ascii(r).count("\\x")
            if len(r) > 0 and (innormal_chars_num * 1.0 / len(r) < 0.5):
                new_r.append(r)
            # else:
            #     print(r)

        new_r = ';'.join(new_r)
        if new_r:
            new_r = "(" + new_r + ")"
        s = s.replace(ori_r, new_r)
        p += len(new_r)

        lrb = s.find("(", p)
    return s