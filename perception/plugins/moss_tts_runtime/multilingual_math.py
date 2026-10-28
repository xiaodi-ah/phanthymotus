"""Conservative mathematical speech for the six non-Chinese contest languages."""
import json,re
from decimal import Decimal
from functools import lru_cache
from pathlib import Path
from . import icu_numbers
from .math_expression import Parser,NUMBER,FRACTIONS

DATA=json.loads(Path(__file__).with_name('multilingual_frontend_data.json').read_text(encoding='utf-8'))
# Scope words are explicit so nested expressions keep their original meaning.
WORDS={
'en':dict(plus='plus',minus='minus',pm='plus or minus',times='times',divide='divided by',equal='equals',lt='less than',gt='greater than',le='less than or equal to',ge='greater than or equal to',ne='not equal to',approx='approximately equals',ratio='to',start='open parenthesis',end='close parenthesis',over='over',power='to the power of',square='squared',cube='cubed',root='square root of',rootindex='root with index',sub='subscript',factorial='factorial',absolute='absolute value of',infinity='infinity',point='point',percent='percent',permille='per mille',permyriad='per ten thousand',and_='and',base='with base',argument='of',per='per'),
'de':dict(plus='plus',minus='minus',pm='plus minus',times='mal',divide='geteilt durch',equal='gleich',lt='kleiner als',gt='größer als',le='kleiner oder gleich',ge='größer oder gleich',ne='ungleich',approx='ungefähr gleich',ratio='zu',start='Klammer auf',end='Klammer zu',over='durch',power='hoch',square='zum Quadrat',cube='hoch drei',root='Quadratwurzel aus',rootindex='Wurzel mit Index',sub='Index',factorial='Fakultät',absolute='Betrag von',infinity='unendlich',point='Komma',percent='Prozent',permille='Promille',permyriad='pro zehntausend',and_='und',base='zur Basis',argument='von',per='pro'),
'fr':dict(plus='plus',minus='moins',pm='plus ou moins',times='fois',divide='divisé par',equal='égale',lt='inférieur à',gt='supérieur à',le='inférieur ou égal à',ge='supérieur ou égal à',ne='différent de',approx='environ égal à',ratio='pour',start='parenthèse ouvrante',end='parenthèse fermante',over='sur',power='à la puissance',square='au carré',cube='au cube',root='racine carrée de',rootindex='racine d’indice',sub='indice',factorial='factorielle',absolute='valeur absolue de',infinity='infini',point='virgule',percent='pour cent',permille='pour mille',permyriad='pour dix mille',and_='et',base='de base',argument='de',per='par'),
'es':dict(plus='más',minus='menos',pm='más o menos',times='por',divide='dividido por',equal='igual a',lt='menor que',gt='mayor que',le='menor o igual que',ge='mayor o igual que',ne='distinto de',approx='aproximadamente igual a',ratio='a',start='abre paréntesis',end='cierra paréntesis',over='sobre',power='elevado a',square='al cuadrado',cube='al cubo',root='raíz cuadrada de',rootindex='raíz de índice',sub='subíndice',factorial='factorial',absolute='valor absoluto de',infinity='infinito',point='coma',percent='por ciento',permille='por mil',permyriad='por diez mil',and_='y',base='de base',argument='de',per='por'),
'ru':dict(plus='плюс',minus='минус',pm='плюс минус',times='умножить на',divide='разделить на',equal='равно',lt='меньше',gt='больше',le='меньше или равно',ge='больше или равно',ne='не равно',approx='приблизительно равно',ratio='к',start='скобка открывается',end='скобка закрывается',over='разделить на',power='в степени',square='в квадрате',cube='в кубе',root='квадратный корень из',rootindex='корень степени',sub='индекс',factorial='факториал',absolute='модуль',infinity='бесконечность',point='запятая',percent='процентов',permille='промилле',permyriad='на десять тысяч',and_='и',base='по основанию',argument='от',per='на'),
'ja':dict(plus='プラス',minus='マイナス',pm='プラスマイナス',times='かける',divide='割る',equal='イコール',lt='小なり',gt='大なり',le='以下',ge='以上',ne='等しくない',approx='ほぼ等しい',ratio='対',start='括弧',end='括弧閉じ',over='分の',power='の',square='の二乗',cube='の三乗',root='平方根',rootindex='累乗根の指数',sub='添字',factorial='の階乗',absolute='絶対値',infinity='無限大',point='点',percent='パーセント',permille='パーミル',permyriad='万分率',and_='と',base='底',argument='の',per='毎')}
OP_KEYS={'+':'plus','-':'minus','−':'minus','±':'pm','*':'times','×':'times','÷':'divide','=':'equal','==':'equal','<':'lt','>':'gt','≤':'le','<=':'le','≥':'ge','>=':'ge','!=':'ne','≠':'ne','≈':'approx','≃':'approx',':':'ratio','∶':'ratio'}
FUNCTIONS={'ja':{'sin':'サイン','cos':'コサイン','tan':'タンジェント','cot':'コタンジェント','sec':'セカント','csc':'コセカント','ln':'自然対数','log':'対数','exp':'指数関数','abs':'絶対値','max':'最大値','min':'最小値'},'ru':{'sin':'синус','cos':'косинус','tan':'тангенс','cot':'котангенс','sec':'секанс','csc':'косеканс','ln':'натуральный логарифм','log':'логарифм','exp':'экспонента','abs':'модуль','max':'максимум','min':'минимум'}}
_GREEK='αβγδεζηθικλμνξοπρστυφχψω'
_GREEK_NAMES={'ja':'アルファ ベータ ガンマ デルタ イプシロン ゼータ イータ シータ イオタ カッパ ラムダ ミュー ニュー クシー オミクロン パイ ロー シグマ タウ ウプシロン ファイ カイ プサイ オメガ'.split(),'ru':'альфа бета гамма дельта эпсилон дзета эта тета йота каппа лямбда мю ню кси омикрон пи ро сигма тау ипсилон фи хи пси омега'.split()}
_COMMANDS={'times':'×','cdot':'×','div':'÷','pm':'±','le':'≤','leq':'≤','ge':'≥','geq':'≥','ne':'≠','neq':'≠','approx':'≈','infty':'∞','left':'','right':''}
_GREEK_COMMANDS=dict(zip('alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi rho sigma tau upsilon phi chi psi omega'.split(),_GREEK))
_CUES={'en':r'math|formula|equation|calculate|ratio','de':r'Mathematik|Formel|Gleichung|berechn|Verhältnis','fr':r'math|formule|équation|calcul|rapport','es':r'matemát|fórmula|ecuación|calcul|proporción','ru':r'математ|формул|уравнен|вычисл|отношен','ja':r'数学|数式|公式|計算|方程式|比率|比は'}
_RATIO={'en':r'ratio','de':r'Verhältnis','fr':r'rapport','es':r'proporción','ru':r'отношен','ja':r'比率|比は'}
_TIME=r'time|clock|Zeit|Uhr|heure|hora|время|時刻|時間'
_TOKENS=re.compile(r'\s+|\\[A-Za-z]+|[A-Za-z]+|[0-9]+(?:[.,][0-9]+)*(?:[eE][+−-]?[0-9]+)?|>=|<=|!=|==|\*\*|[^\s]')
_PROTECTED=re.compile(r'\b[0-9]{4}[-/][0-9]{1,2}[-/][0-9]{1,2}\b|(?:[A-Za-z0-9_.-]+[/\\])+[A-Za-z0-9_.%+-]+|\b[A-Za-z]{2,}[A-Za-z0-9_]*[-_][A-Za-z0-9_.-]+|\b(?:[A-Za-z]+[0-9]+|[0-9]+[A-Za-z]+)(?:\.[A-Za-z0-9]+)+|\b[0-9]+(?:\.[0-9]+){2,}\b')

class Speech:
 def __init__(self,language):
  self.lang=language;self.data=DATA[language];self.words=WORDS[language];self.functions=FUNCTIONS.get(language,self.data['functions']);self.greek=self.data['greek'] or dict(zip(_GREEK,_GREEK_NAMES[language]))
  self.greek={**self.greek,**{k.upper():v for k,v in self.greek.items()}}
  self.unit_pattern=re.compile('(?:'+'|'.join(re.escape(k) for k in sorted(self.data['units'],key=len,reverse=True))+')')
  self.quantity_pattern=re.compile(r'(?<![A-Za-z0-9_.,/\\-])([+−±-]?'+NUMBER.replace(r'\.',r'[.,]')+r')\s*('+self.unit_pattern.pattern+r')(?![A-Za-z0-9_/\\]|\.[A-Za-z0-9_])')
 def join(self,*parts):return (' ' if self.lang!='ja' else '').join(str(p) for p in parts if p!='')
 def w(self,key):return self.words[key]
 def number(self,raw,gender=None):
  raw=raw.replace('−','-').replace(',','.');sign=''
  if raw[0] in '+-±':sign=self.w({'+':'plus','-':'minus','±':'pm'}[raw[0]]);raw=raw[1:]
  parts=re.split('[eE]',raw)
  if len(parts)==2:return self.join(sign,self.number(parts[0]),self.w('times'),self.number('10'),self.w('power'),self.number(parts[1]))+('乗' if self.lang=='ja' else '')
  if not re.fullmatch(NUMBER,raw):raise ValueError('Unsupported number')
  whole,point,tail=raw.partition('.');spoken=icu_numbers.integer(int(whole),self.lang,gender)
  if point:
   if self.lang=='ru':
    n=int(whole);one=n%10==1 and n%100!=11;f=int(tail);fraction_one=f%10==1 and f%100!=11
    spoken=self.join(icu_numbers.integer(n,'ru','feminine'),'целая' if one else 'целых',icu_numbers.integer(f,'ru','feminine'),icu_numbers.ordinal(10**len(tail),'ru',plural=not fraction_one))
   else:spoken=self.join(spoken,self.w('point'),self.join(*(icu_numbers.integer(int(c),self.lang) for c in tail)))
  return self.join(sign,spoken)
 def quantity(self,raw,symbol):
  if self.lang in ('en','ja') and ',' in raw or self.lang in ('de','es') and re.fullmatch(r'[+−±-]?[0-9]{1,3}\.[0-9]{3}',raw):raise ValueError('Ambiguous grouping')
  u=self.data['units'][symbol];gender=u['gender'];spoken=self.number(raw,gender)
  canonical=raw.lstrip('+−±-').replace(',','.');value=Decimal(canonical);n=int(value);dot='.' in canonical or value!=n
  category='other'
  if self.lang=='ru' and not dot:category='one' if n%10==1 and n%100!=11 else 'few' if 2<=n%10<=4 and not 12<=n%100<=14 else 'many'
  elif self.lang=='fr':category='one' if Decimal(canonical)<2 else 'other'
  elif self.lang!='ja' and not dot and n==1:category='one'
  if self.lang=='de' and not dot and n==1 and not re.search('[eE]',raw):spoken=self.join(self.w('minus') if raw[0] in '-−' else self.w('plus') if raw[0]=='+' else self.w('pm') if raw[0]=='±' else '', 'eine' if gender=='feminine' else 'ein')
  if self.lang=='es' and not dot and gender=='masculine':spoken=re.sub(r'\buno$','un',spoken);spoken=re.sub(r'veintiuno$','veintiún',spoken)
  return u['patterns'].get(category,u['patterns']['other']).replace('{0}',spoken).strip()
 def render(self,node):
  kind,*a=node;r=self.render;w=self.w;j=self.join
  if kind=='number':return self.number(a[0])
  if kind=='quantity':return self.quantity(*a)
  if kind=='greek':return self.greek[a[0]]
  if kind=='variable':return (self.data.get('math_letters') or self.data['letters']).get(a[0],a[0])
  if kind=='infinity':return w('infinity')
  if kind=='sequence':return j(r(a[0]),w('and_'),r(a[1]))
  if kind=='binary':
   op,left,right=a
   if op=='/':return self.fraction(left,right)
   return j(r(left),w(OP_KEYS[op]),r(right))
  if kind=='unary':return j(w({'+':'plus','-':'minus','−':'minus','±':'pm'}[a[0]]),r(a[1]))
  if kind=='group':return j(w('start'),r(a[0]),w('end'))
  if kind=='fraction':return self.fraction(*a)
  if kind=='power':
   base,exp=a
   if exp in (('number','2'),('number','3')):return j(r(base),w('square' if exp[1]=='2' else 'cube'))
   return j(r(base),w('power'),r(exp))+('乗' if self.lang=='ja' else '')
  if kind=='subscript':return j(r(a[0]),w('sub'),r(a[1]))
  if kind=='percent':return j(r(a[0]),w({'%':'percent','‰':'permille','‱':'permyriad'}[a[1]]))
  if kind=='factorial':return j(r(a[0]),w('factorial'))
  if kind=='absolute':return j(w('absolute'),r(a[0]))
  if kind=='root':return j(w('root'),r(a[1])) if a[0] is None else j(w('rootindex'),r(a[0]),w('argument'),r(a[1]))
  if kind=='function':
   name,args,base=a;return j(self.functions[name],j(w('base'),r(base)) if base else '',w('argument'),j(*(r(x) if i==0 else j(w('and_'),r(x)) for i,x in enumerate(args))))
  raise ValueError('Unknown math node')
 def fraction(self,top,bottom):
  # Explicit parentheses preserve scope in compound numerators and denominators.
  def scoped(node):return self.render(node if node[0] in ('number','variable','greek','group') else ('group',node))
  if self.lang=='ja':return self.join(scoped(bottom),self.w('over'),scoped(top))
  return self.join(scoped(top),self.w('over'),scoped(bottom))
 def expression(self,value):
  def command(m):
   name=m[1]
   if name in _COMMANDS:return _COMMANDS[name]
   if name in self.functions or name in ('sqrt','frac'):return name
   if name in _GREEK_COMMANDS:return _GREEK_COMMANDS[name]
   raise ValueError('Unknown LaTeX command')
  value=re.sub(r'\\([A-Za-z]+)',command,value)
  return self.render(Parser(value,self.unit_pattern,OP_KEYS,self.functions,self.greek,decimal_comma=self.lang in ('de','fr','es','ru')).parse())
 def normalize(self,text):
  def part(value):
   tokens=[t for t in _TOKENS.finditer(value) if not t[0].isspace()];runs=[];start=None;end=None;invalid=False
   known=set(self.functions)|set(self.data['units'])|{'sqrt','frac'}
   mathchars=set('0123456789_αβγδεζηθικλμνξοπρςστυφχψωΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩ∞√∛∜±+−-*/×÷^=<>≥≤≠≈≃:∶%‰‱!|(),;\\{}[]$²³⁰¹⁴⁵⁶⁷⁸⁹₀₁₂₃₄₅₆₇₈₉℃℉°µμΩ')|set(FRACTIONS)
   def flush():
    nonlocal start,end,invalid
    if start is not None:runs.append((start,end,invalid))
    start=end=None;invalid=False
   for i,t in enumerate(tokens):
    token=t[0]
    valid=token in known or token.startswith('\\') or len(token)==1 and (0x2200<=ord(token)<=0x22ff or 0x2a00<=ord(token)<=0x2aff or token in mathchars or token.isascii() and token.isalpha()) or token[0].isdigit() or token in ('>=','<=','!=','==','**')
    if valid:
     if start is None:
      start=t.start();invalid=bool(token!=':' and start and value[start-1].isascii() and (value[start-1].isalnum() or value[start-1] in '_./\\-'))
     end=t.end()
    else:
     following=tokens[i+1][0] if i+1<len(tokens) else ''
     prev=value[start:end].rstrip() if start is not None else ''
     if start is not None and (t.start()==end and token[0].isascii() and token[0].isalpha() or prev[-1:] in '=+*/^<>−-'):
      end=t.end();invalid=True
     elif following in ('=','<','>','==',':='):
      flush();start=t.start();end=t.end();invalid=True
     else:flush()
   flush();result=[];last=0
   for start,end,bad in runs:
    original=value[start:end]
    if original.startswith(':'):
     start+=1
     while start<end and value[start].isspace():start+=1
     original=value[start:end]
    candidate=original.rstrip('.,;');suffix=original[len(candidate):];rendered=original
    context=re.split(r'[.!?。！？;；\n]',value[max(0,start-64):start])[-1]
    force=candidate.startswith('$') and candidate.endswith('$') and len(candidate)>2
    if force:candidate=candidate[1:-1]
    cue=bool(re.search(_CUES[self.lang],context,re.I))
    distinctive=bool(re.search(r'[×÷√∛∜±²³π∞‰‱]|\^|\*\*|\\(?:frac|sqrt|times)|[0-9][eE][+−-]?[0-9]',candidate)) or any(c in FRACTIONS for c in candidate)
    relation=bool(re.search(r'\s[=<>]|[=<>]\s|[≥≤≠≈≃]',candidate))
    function=any(re.search(r'\b'+k+r'(?=\s*[(_])',candidate) for k in (*self.functions,'sqrt'))
    arithmetic=bool(re.search(r'[0-9]\s*\+\s*[0-9]',candidate))
    ratio=bool(re.search(_RATIO[self.lang],context,re.I)) and not re.search(_TIME,re.split('[,，]',context)[-1],re.I)
    eligible=force or cue or distinctive or relation or function or arithmetic or bool(self.quantity_pattern.fullmatch(candidate))
    if ':' in candidate and not (force or ratio):eligible=False
    if re.fullmatch(r'[A-Za-z \t]+',candidate):eligible=False
    if re.fullmatch(NUMBER,candidate) and not force:eligible=False
    if len(candidate)>512 or bad:eligible=False
    if eligible:
     try:rendered=self.expression(candidate)+suffix
     except (ValueError,RecursionError,KeyError):pass
    # Separate quantities in prose still work when no compound expression is present.
    if rendered==original and not bad and not re.search(r'[=<>^×÷+*/]|\\|\*\*',candidate.replace('m/s','').replace('km/h','')):
     def quantity(m):
      try:return self.quantity(m[1],m[2])
      except ValueError:return m[0]
     rendered=self.quantity_pattern.sub(quantity,original)
    result.extend((value[last:start],rendered));last=end
   result.append(value[last:]);return ''.join(result)
  pieces=[];last=0
  for match in _PROTECTED.finditer(text):
   # Unit rates are not file paths.
   if self.quantity_pattern.fullmatch(match[0]) or re.fullmatch(r'[0-9]+/[0-9]+',match[0]) or any(match[0].startswith(k+'_') for k in self.functions):continue
   pieces.extend((part(text[last:match.start()]),match[0]));last=match.end()
  pieces.append(part(text[last:]));return ''.join(pieces)

@lru_cache(maxsize=6)
def speech(language):return Speech(language)
def normalize(text,language):return speech(language).normalize(text)
