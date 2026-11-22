"""Read pads (component, name, net), components, vias and tracks straight from a saved .PcbDoc
(OLE compound file), independent of Altium. Record layout follows KiCad's Altium importer
(altium_parser_pcb.cpp: APAD6, ANET6, ACOMPONENT6, AVIA6, ATRACK6)."""
import struct, sys, json, olefile

path = sys.argv[1]
ole = olefile.OleFileIO(path)


def text_records(stream):
    data = ole.openstream(stream).read()
    out, i = [], 0
    while i + 4 <= len(data):
        n = struct.unpack_from('<I', data, i)[0]; i += 4
        raw = data[i:i + n]; i += n
        rec = {}
        for kv in raw.decode('latin-1').strip('\x00').split('|'):
            if '=' in kv:
                k, v = kv.split('=', 1); rec[k.upper()] = v
        out.append(rec)
    return out


nets = [r.get('NAME', '') for r in text_records('Nets6/Data')]
comps = text_records('Components6/Data')
cname = [c.get('SOURCEDESIGNATOR', c.get('DESIGNATOR', '?')) for c in comps]

data = ole.openstream('Pads6/Data').read()
i, pads = 0, []
while i < len(data):
    rtype = data[i]; i += 1
    if rtype != 2:
        raise SystemExit('bad pad record type %d at %d' % (rtype, i))
    n1 = struct.unpack_from('<I', data, i)[0]; i += 4
    sl = data[i]; name = data[i + 1:i + 1 + sl].decode('latin-1'); i += n1
    for _ in range(3):
        n = struct.unpack_from('<I', data, i)[0]; i += 4 + n
    n5 = struct.unpack_from('<I', data, i)[0]; i += 4
    rec = data[i:i + n5]; i += n5
    layer = rec[0]
    net = struct.unpack_from('<H', rec, 3)[0]
    comp = struct.unpack_from('<H', rec, 7)[0]
    x, y = struct.unpack_from('<ii', rec, 13)
    # optional subrecord 6 (size and shape)
    if i + 4 <= len(data) and data[i] != 2:
        n6 = struct.unpack_from('<I', data, i)[0]; i += 4 + n6
    pads.append({'comp': cname[comp] if comp != 0xFFFF and comp < len(cname) else '',
                 'name': name, 'net': nets[net] if net != 0xFFFF and net < len(nets) else '',
                 'x_mil': x / 10000.0, 'y_mil': y / 10000.0})


def count_records(stream, rtype):
    try:
        d = ole.openstream(stream).read()
    except OSError:
        return None
    j, k = 0, 0
    while j < len(d):
        if d[j] != rtype:
            break
        n = struct.unpack_from('<I', d, j + 1)[0]
        j += 5 + n; k += 1
    return k


res = {'nets': len(nets), 'components': len(comps), 'pads': len(pads),
       'vias': count_records('Vias6/Data', 3), 'tracks': count_records('Tracks6/Data', 4),
       'component_comments': {c.get('SOURCEDESIGNATOR', '?'): c.get('COMMENT', c.get('SOURCECOMMENT', '')) for c in comps if c.get('SOURCEDESIGNATOR') in ('U6', 'C27')}}
json.dump({'summary': res, 'pads': pads}, open(sys.argv[2], 'w', encoding='utf-8'), indent=1)
print(json.dumps(res, indent=1))
