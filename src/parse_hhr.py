import re

match_re = re.compile(r'(\d+) +(.+) +(\d+) +\((\d+)\)')
def parse_match(rest):
    start, data, end, total = match_re.findall(rest)[0]
    return int(start), data.rstrip(), int(end), int(total)

def parse_scores(line):
    scores = { key: float(value.replace('%', '')) for key, value in (pair.split('=') for pair in line.split()) }
    return scores

def parse_line(line):
    fields = line.split(maxsplit = 2) or [ "" ]
    if len(fields) == 1: 
        fields.append("")
    if len(fields) == 2:
        fields.append("")
    return fields

def format_record(query, ss_pred, ss_dssp, alignment, consensus, coord_from, coord_to, lens, template, description, match, confidence):
    record = {
        "query": {
            "name": query,
            "ss_pred": ss_pred["Q"],
            "ss_dssp": ss_dssp["Q"],
            "alignment": alignment["Q"],
            "consensus": consensus["Q"],
            "coords": [ coord_from["Q"], coord_to["Q"] ],
            "len": lens["Q"]
        },
        "template": {
            "name": template,
            "description": description,
            "ss_pred": ss_pred["T"],
            "ss_dssp": ss_dssp["T"],
            "alignment": alignment["T"],
            "consensus": consensus["T"],
            "coords": [ coord_from["T"], coord_to["T"] ],
            "len": lens["T"]
        },
        "match": match,
        "confidence": confidence
    }
    return record

def read_hhr(file):
    next_line_match = False
    template = scores = None
    for line in file:
        line = line.replace('\x00', '')
        key, value, rest = parse_line(line) 
        if key == "Query":
            if template:
                record = format_record(query, ss_pred, ss_dssp, alignment, consensus, coord_from, coord_to, lens, template, description, match, confidence)
                yield {**record, **scores}
            query = value
            template = scores = None
            next_line_match = False
        elif key == "No":
            if template:
                record = format_record(query, ss_pred, ss_dssp, alignment, consensus, coord_from, coord_to, lens, template, description, match, confidence)
                yield {**record, **scores}
                template = None
        elif key.startswith("Probab="):
            scores = parse_scores(line)
        elif key.startswith(">"):
            template = key[1:]
            description = f"{value} {rest}".strip()
            ss_pred   = { "Q": "", "T": "" }
            ss_dssp   = { "Q": "", "T": "" }
            consensus = { "Q": "", "T": "" }
            alignment = { "Q": "", "T": "" }
            coord_to = { "Q": None, "T": None }
            coord_from = { "Q": None, "T": None }
            lens = { "Q": None, "T": None }
            confidence = ""
            match = ""
        elif key == "Q" or key == "T":
            if value == "ss_pred":
                ss_pred[key] += rest.strip()
            elif value == "ss_dssp":
                ss_dssp[key] += rest.strip()
            elif value == "Consensus":
                start, cons, end, total = parse_match(rest)
                consensus[key] += cons
                if key == "Q":
                    next_line_match = True
                    match_start = line.rindex(cons)
                    match_stop  = match_start + len(cons)
            else:
                start, aln, end, total = parse_match(rest)
                alignment[key] += aln
                lens[key] = total
                coord_to[key] = end
                if coord_from[key] is None:
                    coord_from[key] = start
        elif key == "Confidence":
            confidence += line[match_start:match_stop]
        elif next_line_match:
            match += line[match_start:match_stop]
            next_line_match = False
    if template:
        record = format_record(query, ss_pred, ss_dssp, alignment, consensus, coord_from, coord_to, lens, template, description, match, confidence)
        yield {**record, **scores}
