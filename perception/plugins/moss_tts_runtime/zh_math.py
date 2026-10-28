"""Bounded Chinese math grammar; public source data is recorded beside the runtime."""
import json
import re
from pathlib import Path

DATA=json.loads(Path(__file__).with_name('zh_frontend_data.json').read_text(encoding='utf-8'))
UNITS=DATA['units']
OPERATORS=DATA['operators']
FUNCTIONS=DATA['functions']
GREEK=DATA['greek']
_DIGITS='零一二三四五六七八九'
_WIDTH={code:code-0xFEE0 for code in range(0xFF01,0xFF5F)}
_NUMBER=r'(?:0|[1-9][0-9]{0,15})(?:\.[0-9]{1,8})?'
_UNIT='(?:'+'|'.join(re.escape(k) for k in sorted(UNITS,key=len,reverse=True))+')'
_UNIT_RE=re.compile(_UNIT+r'(?:/'+_UNIT+r')?(?![A-Za-z0-9_/\\]|\.[A-Za-z0-9_])')
_QUANTITY=re.compile(r'(?<![A-Za-z0-9_.,/\\-])([+−±-]?'+_NUMBER+r')\s*('+_UNIT+r'(?:/'+_UNIT+r')?)(?![A-Za-z0-9_/\\]|\.[A-Za-z0-9_])')
_RANGE=re.compile(r'(?<![A-Za-z0-9_.,/\\-])([+−-]?'+_NUMBER+r')\s*('+_UNIT+r')?\s*([~～–至到-])\s*([+−-]?'+_NUMBER+r')\s*('+_UNIT+r')(?![A-Za-z0-9_/\\]|\.[A-Za-z0-9_])')
_TOKEN=re.compile(r'\s+|'+_NUMBER+r'(?:[eE][+−-]?[0-9]{1,3})?|\\[A-Za-z]+|[A-Za-z]+|[α-ωΑ-Ω]|>=|<=|!=|==|\*\*|[^\s]')
# Consume the whole candidate, including unsupported syntax, so a failed parse cannot rewrite a subexpression.
_SPAN=re.compile(r'[A-Za-z0-9_０-９Ａ-Ｚａ-ｚ＋－＊／＝＜＞＾＿．％：｜！（）［］｛｝\u0370-\u03ff\u2200-\u22ff\u2a00-\u2affα-ωΑ-Ωπ∞√∛∜∑∫∏∂∇±+−\-*/×÷^=<>≥≤≠≈≃~～:∶%‰‱!|().,\\{}\[\]$²³⁰¹⁴⁵⁶⁷⁸⁹₀-₉¼½¾⅐⅑⅒⅓⅔⅕⅖⅗⅘⅙⅚⅛⅜⅝⅞℃℉°µμΩ]+(?:[ \t]+[A-Za-z0-9_０-９Ａ-Ｚａ-ｚ＋－＊／＝＜＞＾＿．％：｜！（）［］｛｝\u0370-\u03ff\u2200-\u22ff\u2a00-\u2affα-ωΑ-Ωπ∞√∛∜∑∫∏∂∇±+−\-*/×÷^=<>≥≤≠≈≃~～:∶%‰‱!|().,\\{}\[\]$²³⁰¹⁴⁵⁶⁷⁸⁹₀-₉¼½¾⅐⅑⅒⅓⅔⅕⅖⅗⅘⅙⅚⅛⅜⅝⅞℃℉°µμΩ]+)*')
_CUE=re.compile(r'数学|算式|公式|计算(?!机)|等式|方程|不等式|分数|分之一|比例|比值|比分|比为|科学计数|数值|平方|立方|根号|阶乘|对数|正弦|余弦|函数')
_SUPER=str.maketrans('⁰¹²³⁴⁵⁶⁷⁸⁹','0123456789')
_SUB=str.maketrans('₀₁₂₃₄₅₆₇₈₉','0123456789')
_FRACTIONS={'¼':(1,4),'½':(1,2),'¾':(3,4),'⅐':(1,7),'⅑':(1,9),'⅒':(1,10),'⅓':(1,3),'⅔':(2,3),'⅕':(1,5),'⅖':(2,5),'⅗':(3,5),'⅘':(4,5),'⅙':(1,6),'⅚':(5,6),'⅛':(1,8),'⅜':(3,8),'⅝':(5,8),'⅞':(7,8)}
_COMMANDS={'times':'×','cdot':'×','div':'÷','pm':'±','le':'≤','leq':'≤','ge':'≥','geq':'≥','ne':'≠','neq':'≠','approx':'≈','infty':'∞','left':'','right':''}


def integer(value):
    if not 0<=value<10**16:raise ValueError('Chinese number outside supported range')
    if value==0:return '零'
    for base,unit in ((10**8,'亿'),(10**4,'万')):
        if value>=base:
            high,low=divmod(value,base)
            tail=integer(low) if low else ''
            if 10<=low<20:tail='一'+tail
            return integer(high)+unit+('零' if 0<low<base//10 else '')+tail
    result='';zero=False
    for base,unit in ((1000,'千'),(100,'百'),(10,'十'),(1,'')):
        digit,value=divmod(value,base)
        if digit:
            if zero:result+='零'
            result+=_DIGITS[digit]+unit;zero=False
        elif result and value:zero=True
    return result[1:] if result.startswith('一十') else result


def number(raw):
    sign=''
    if raw[0] in '+−±-':sign={'+':'正','-':'负','−':'负','±':'正负'}[raw[0]];raw=raw[1:]
    if not re.fullmatch(_NUMBER,raw):raise ValueError('Unsupported or ambiguous number')
    whole,point,fraction=raw.partition('.')
    return sign+integer(int(whole))+('点'+''.join(_DIGITS[int(c)] for c in fraction) if point else '')


def unit(raw):
    parts=raw.split('/')
    return UNITS[parts[0]]+('每'+UNITS[parts[1]] if len(parts)==2 else '')


def quantity(raw,symbol):
    spoken=number(raw)
    if symbol not in ('%','％','‰','‱'):return spoken+unit(symbol)
    sign={'-':'负','−':'负','+':'正','±':'正负'}.get(raw[0],'')
    return sign+{'%':'百分之','％':'百分之','‰':'千分之','‱':'万分之'}[symbol]+spoken[len(sign):]


def scoped(value):
    if value.startswith('左括号'):
        depth=0
        for match in re.finditer('左括号|右括号',value):
            depth+=1 if match[0]=='左括号' else -1
            if depth==0:
                if match.end()==len(value):return value
                break
    composite=re.search('加|减|乘以|除以|等于|小于|大于|分之|次方|的平方|的立方|的绝对值|根号',value) or any(value.startswith(name) for name in FUNCTIONS.values())
    return '左括号'+value+'右括号' if composite else value


class Expression:
    def __init__(self,text):
        self.tokens=[m[0] for m in _TOKEN.finditer(text) if not m[0].isspace()]
        if len(self.tokens)>128:raise ValueError('Math token limit')
        self.i=0;self.depth=0

    def peek(self):return self.tokens[self.i] if self.i<len(self.tokens) else ''
    def take(self,expected=None):
        token=self.peek()
        if not token or expected is not None and token!=expected:raise ValueError('Invalid expression')
        self.i+=1;return token

    def parse(self):
        value=self.relation()
        while self.peek()==',':self.take(',');value+='，'+self.relation()
        if self.peek():raise ValueError('Unsupported expression suffix')
        return value

    def relation(self):
        result=self.sum()
        while self.peek() in OPERATORS and self.peek() not in ('+','-','−','*','×','/','÷'):
            operator=self.take();result+=OPERATORS[operator]+self.sum()
        return result

    def sum(self):
        result=self.product()
        while self.peek() in ('+','-','−','±'):
            operator=self.take();result+={'+':'加','-':'减','−':'减','±':'正负'}[operator]+self.product()
        return result

    def product(self):
        result=self.power()
        while self.peek():
            token=self.peek()
            if token in ('*','×','÷','/'):
                self.take();right=self.power()
                result=(scoped(right)+'分之'+scoped(result)) if token=='/' else result+('除以' if token=='÷' else '乘以')+right
            elif token in GREEK or re.fullmatch('[A-Za-z]',token) or token in ('(', '{', '√', '∛', '∜', 'sqrt'):
                result+='乘以'+self.power()
            else:break
        return result

    def power(self):
        self.depth+=1
        if self.depth>16:raise ValueError('Math nesting limit')
        try:
            if self.peek() in ('+','-','−','±'):
                sign=self.take();return {'+':'正','-':'负','−':'负','±':'正负'}[sign]+self.power()
            result=self.atom()
            while self.peek() in ('²','³','⁰','¹','⁴','⁵','⁶','⁷','⁸','⁹','^','**','_','%','‰','‱','!') or self.peek() and self.peek()[0] in '₀₁₂₃₄₅₆₇₈₉':
                token=self.take()
                if token in ('^','**'):
                    exponent=self.power();result=scoped(result)+'的'+('平方' if exponent=='二' else '立方' if exponent=='三' else scoped(exponent)+'次方')
                elif token=='_':result+='下标'+self.atom()
                elif token in ('%','‰','‱'):result={'%':'百分之','‰':'千分之','‱':'万分之'}[token]+result
                elif token=='!':result+='的阶乘'
                elif token[0] in '₀₁₂₃₄₅₆₇₈₉':
                    while self.peek() and self.peek()[0] in '₀₁₂₃₄₅₆₇₈₉':token+=self.take()
                    result+='下标'+integer(int(token.translate(_SUB)))
                else:
                    digits=token
                    while self.peek() and self.peek()[0] in '⁰¹²³⁴⁵⁶⁷⁸⁹':digits+=self.take()
                    exponent=integer(int(digits.translate(_SUPER)))
                    result=scoped(result)+'的'+('平方' if exponent=='二' else '立方' if exponent=='三' else exponent+'次方')
            return result
        finally:self.depth-=1

    def group(self,opening):
        closing={'(':')','{':'}','[':']','|':'|'}[opening]
        value=self.relation();self.take(closing)
        return value

    def atom(self):
        token=self.take()
        if token in ('(', '{'):
            value=self.group(token)
            return value if token=='{' else '左括号'+value+'右括号'
        if token=='|':return self.group('|')+'的绝对值'
        if token in ('√','∛','∜','sqrt'):
            index={'∛':'三','∜':'四'}.get(token)
            if self.peek()=='[':self.take('[');index=self.group('[')
            opening=self.peek();value=self.atom() if opening in ('(', '{') else self.power()
            if opening!='(':value=scoped(value)
            return '根号'+value if index is None else value+'的'+('立方根' if index=='三' else index+'次方根')
        if token=='frac':
            self.take('{');numerator=self.group('{');self.take('{');denominator=self.group('{')
            return scoped(denominator)+'分之'+scoped(numerator)
        if token in FUNCTIONS:
            base=None
            if token=='log' and self.peek()=='_':self.take('_');base=self.atom()
            self.take('(');arguments=[self.relation()]
            while self.peek()==',':self.take(',');arguments.append(self.relation())
            self.take(')')
            if token in ('max','min'):return '与'.join(scoped(a) for a in arguments)+'的'+FUNCTIONS[token]
            if len(arguments)!=1:raise ValueError('Unexpected function arguments')
            argument=scoped(arguments[0])
            if token=='abs':return argument+'的绝对值'
            return '以'+scoped(base)+'为底'+argument+'的对数' if base else FUNCTIONS[token]+argument
        if re.fullmatch(_NUMBER+r'(?:[eE][+−-]?[0-9]{1,3})?',token):
            parts=re.split('[eE]',token)
            spoken=number(parts[0])
            if len(parts)==2:spoken+='乘以十的'+number(parts[1])+'次方'
            # Join a complete unit token sequence before interpreting implicit algebraic multiplication.
            for count in range(min(5,len(self.tokens)-self.i),0,-1):
                candidate=''.join(self.tokens[self.i:self.i+count])
                if _UNIT_RE.fullmatch(candidate):self.i+=count;return quantity(parts[0],candidate) if len(parts)==1 else spoken+unit(candidate)
            return spoken
        if token in GREEK:return GREEK[token]
        if token=='∞':return '无穷大'
        if re.fullmatch('[A-Za-z]',token):return token.upper()
        if token in _FRACTIONS:
            numerator,denominator=_FRACTIONS[token];return integer(denominator)+'分之'+integer(numerator)
        raise ValueError('Unsupported math token')


def normalize(text):
    def math(match):
        original=match[0];value=original.rstrip('.');suffix=original[len(value):];value=value.translate(_WIDTH)
        label=''
        if value.startswith(':'):label=original[0];value=value[1:].lstrip()
        if len(value)>512:return original
        force=value.startswith('$') and value.endswith('$') and len(value)>2
        if force:value=value.strip('$')
        elif value.startswith(('\\(','\\[')) and value[-2:] in ('\\)','\\]'):value=value[2:-2];force=True
        elif '$' in value:return original
        context=re.split(r'[。；;！？!?\n]',text[max(0,match.start()-64):match.start()])[-1][-32:]
        local_context=re.split('[，,]',context)[-1]
        ratio_context=bool(re.search(r'比例|比值|比分|比为',local_context)) and not re.search(r'时间|时刻|时钟|时长|日期',local_context)
        cue=bool(_CUE.search(context))
        if not force and re.fullmatch(_NUMBER,value):
            if text[match.end():match.end()+1] in ('年','月','日','号') or not re.search(r'数值(?:为|是)?[ \t]*$',context):return original
        # Slash-only dates/paths, clocks and unspaced code assignments stay literal without explicit math context.
        relation=bool(re.search(r'\s[=<>]|[=<>]\s|[≥≤≠≈≃]',value))
        distinctive=bool(re.search(r'[×÷√∛∜π∞±²³⁰¹⁴⁵⁶⁷⁸⁹‰‱]|\^|\*\*|\\(?:frac|sqrt|times|div|pi)\b|[0-9][eE][+−-]?[0-9]',value)) or any(c in _FRACTIONS for c in value)
        distinctive=distinctive or any(re.search(r'\b'+re.escape(name)+r'(?=\s*[(_])',value) for name in FUNCTIONS) or value.startswith('|') and value.endswith('|')
        arithmetic=bool(re.search(r'[0-9]\s*\+\s*[0-9]',value))
        if not (force or cue or relation or distinctive or arithmetic or _QUANTITY.fullmatch(value)):return original
        if not force and re.fullmatch('[A-Za-z \t]+',value):return original
        function_call=any(re.search(r'\b'+re.escape(name)+r'\s*\(',value) for name in FUNCTIONS)
        if re.search(r'[0-9],[0-9]{3}(?:\D|$)',value) and not function_call or re.search(r'[0-9]\.[0-9]+\.',value):return original
        if ':' in value and not (force or ratio_context):return original
        if re.search(r'\b[0-9]{4}[-/][0-9]{1,2}[-/][0-9]{1,2}\b',value):return original
        if re.search(r'\b[0-9]{1,2}:[0-9]{2}\b',value) and not (force or ratio_context):return original
        def command(m):
            name=m[1]
            if name in _COMMANDS:return _COMMANDS[name]
            if name in ('sqrt','frac') or name in FUNCTIONS:return name
            greek=DATA['greek_commands'].get(name)
            if greek:return greek
            raise ValueError('Unsupported LaTeX command')
        try:
            value=re.sub(r'\\([A-Za-z]+)',command,value)
            return label+Expression(value).parse()+suffix
        except (ValueError,RecursionError):return original
    def ranges(match):
        left,u1,connector,right,u2=match.groups()
        if connector=='-' and not re.search(r'范围|区间|电压|电流|温度|速度|功率|频率|长度|距离|质量|重量|容量',text[max(0,match.start()-12):match.start()]):return match[0]
        if u2 in ('%','％','‰','‱'):return quantity(left,u1 or u2)+'到'+quantity(right,u2)
        return number(left)+(unit(u1) if u1 and u1!=u2 else '')+'到'+number(right)+unit(u2)
    text=_RANGE.sub(ranges,text)
    text=_SPAN.sub(math,text)
    return _QUANTITY.sub(lambda m:quantity(m[1],m[2]),text)
