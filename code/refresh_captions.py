"""Reapply stored editorial caption metadata after presentation regeneration."""
from pathlib import Path
import json,re

def main(root):
    t=Path(root)/'manuscript';metadata=json.loads((t/'caption_metadata.json').read_text())
    for name in {v['file'] for v in metadata.values()}:
        p=t/name;s=p.read_text();edits=[]
        for env in re.finditer(r'\\begin\{(figure\*?|table\*?)\}[\s\S]*?\\end\{\1\}',s):
            b=env.group();lm=re.search(r'\\label\{([^}]+)\}',b);cm=re.search(r'\\caption(?:\[[^\]]*\])?\{',b)
            if not(lm and cm) or lm[1] not in metadata:continue
            start=cm.end();i=start;depth=1
            while depth:
                if b[i]=='{' and b[i-1]!='\\':depth+=1
                elif b[i]=='}' and b[i-1]!='\\':depth-=1
                i+=1
            edits.append((env.start()+start,env.start()+i-1,metadata[lm[1]]['caption']))
        for a,b,v in reversed(edits):s=s[:a]+v+s[b:]
        p.write_text(s)
    # Keep the exported caption catalogue in step with the manuscript without
    # touching any chart, curve point, or performance record.
    figures={}
    for name in {v['file'] for v in metadata.values()}:
        for b in re.findall(r'\\begin\{figure\*?\}([\s\S]*?)\\end\{figure\*?\}',(t/name).read_text()):
            gm=re.search(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}',b);lm=re.search(r'\\label\{([^}]+)\}',b)
            if gm and lm and lm[1] in metadata:figures[Path(gm[1]).stem]=metadata[lm[1]]['caption']
    f=Path(root)/'figures_r5/figure_catalogue.json'
    cat=json.loads(f.read_text())
    for row in cat:row['caption']=figures[row['name']]
    f.write_text(json.dumps(cat,indent=2)+'\n')
    (f.parent/'CAPTIONS.md').write_text('\n\n'.join('## '+r['name']+'\n\n'+r['caption']+'\n\nSource: `'+r['source']+'`' for r in cat)+'\n')

if __name__=='__main__':main(Path(__file__).resolve().parents[1])
