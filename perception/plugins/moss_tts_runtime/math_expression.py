"""Bounded, non-evaluating math syntax shared by language-specific speech renderers."""
import re

NUMBER=r'(?:0|[1-9][0-9]{0,15})(?:\.[0-9]{1,8})?'
FRACTIONS={'¼':(1,4),'½':(1,2),'¾':(3,4),'⅐':(1,7),'⅑':(1,9),'⅒':(1,10),'⅓':(1,3),'⅔':(2,3),'⅕':(1,5),'⅖':(2,5),'⅗':(3,5),'⅘':(4,5),'⅙':(1,6),'⅚':(5,6),'⅛':(1,8),'⅜':(3,8),'⅝':(5,8),'⅞':(7,8)}
SUPER=str.maketrans('⁰¹²³⁴⁵⁶⁷⁸⁹','0123456789');SUB=str.maketrans('₀₁₂₃₄₅₆₇₈₉','0123456789')


class Parser:
    def __init__(self,text,unit_pattern,operators,functions,greek,decimal_comma=False):
        self.units=unit_pattern;self.operators=operators;self.functions=functions;self.greek=greek
        self.number=NUMBER.replace(r'\.',r'[.,]') if decimal_comma else NUMBER
        pattern=re.compile(r'\s+|'+self.number+r'(?:[eE][+−-]?[0-9]{1,3})?|\\[A-Za-z]+|[A-Za-z]+|[α-ωΑ-Ω]|>=|<=|!=|==|\*\*|[^\s]')
        self.tokens=[];stack=[]
        for match in pattern.finditer(text):
            token=match[0]
            if token.isspace():continue
            if token=='(':
                multi=bool(self.tokens and self.tokens[-1] in ('max','min'));depth=1;semicolon=False
                for char in text[match.end():]:
                    if char=='(':depth+=1
                    elif char==')':
                        depth-=1
                        if not depth:break
                    elif char==';' and depth==1:semicolon=True
                stack.append(multi and not semicolon)
            if decimal_comma and ',' in token and stack and stack[-1]:
                left,right=token.split(',',1);self.tokens.extend([left,',',right])
            else:self.tokens.append(token)
            if token==')' and stack:stack.pop()
        if len(self.tokens)>128:raise ValueError('Math token limit')
        self.i=0;self.depth=0
    def peek(self):return self.tokens[self.i] if self.i<len(self.tokens) else ''
    def take(self,expected=None):
        token=self.peek()
        if not token or expected is not None and token!=expected:raise ValueError('Invalid expression')
        self.i+=1;return token
    def parse(self):
        result=self.relation()
        while self.peek() in (',',';'):
            self.take();result=('sequence',result,self.relation())
        if self.peek():raise ValueError('Unsupported expression suffix')
        return result
    def relation(self):
        result=self.sum()
        while self.peek() in self.operators and self.peek() not in ('+','-','−','*','×','/','÷'):
            operator=self.take();result=('binary',operator,result,self.sum())
        return result
    def sum(self):
        result=self.product()
        while self.peek() in ('+','-','−','±'):
            operator=self.take();result=('binary',operator,result,self.product())
        return result
    def product(self):
        result=self.power()
        while self.peek():
            token=self.peek()
            if token in ('*','×','÷','/'):
                self.take();result=('binary',token,result,self.power())
            elif token in self.greek or re.fullmatch('[A-Za-z]',token) or token in ('(', '{','√','∛','∜','sqrt'):
                result=('binary','×',result,self.power())
            else:break
        return result
    def power(self):
        self.depth+=1
        if self.depth>16:raise ValueError('Math nesting limit')
        try:
            if self.peek() in ('+','-','−','±'):
                sign=self.take();return ('unary',sign,self.power())
            result=self.atom()
            while self.peek() in ('²','³','⁰','¹','⁴','⁵','⁶','⁷','⁸','⁹','^','**','_','%','‰','‱','!') or self.peek() and self.peek()[0] in '₀₁₂₃₄₅₆₇₈₉':
                token=self.take()
                if token in ('^','**'):result=('power',result,self.power())
                elif token=='_':result=('subscript',result,self.atom())
                elif token in ('%','‰','‱'):result=('percent',result,token)
                elif token=='!':result=('factorial',result)
                elif token[0] in '₀₁₂₃₄₅₆₇₈₉':
                    while self.peek() and self.peek()[0] in '₀₁₂₃₄₅₆₇₈₉':token+=self.take()
                    result=('subscript',result,('number',token.translate(SUB)))
                else:
                    while self.peek() and self.peek()[0] in '⁰¹²³⁴⁵⁶⁷⁸⁹':token+=self.take()
                    result=('power',result,('number',token.translate(SUPER)))
            return result
        finally:self.depth-=1
    def group(self,opening):
        value=self.relation();self.take({'(':')','{':'}','[':']','|':'|'}[opening]);return value
    def atom(self):
        token=self.take()
        if token in ('(', '{'):
            value=self.group(token);return ('group',value) if token=='(' else value
        if token=='|':return ('absolute',self.group('|'))
        if token in ('√','∛','∜','sqrt'):
            index={'∛':('number','3'),'∜':('number','4')}.get(token)
            if self.peek()=='[':self.take('[');index=self.group('[')
            opening=self.peek();value=self.atom() if opening in ('(', '{') else self.power()
            return ('root',index,value,opening=='(')
        if token=='frac':
            self.take('{');top=self.group('{');self.take('{');bottom=self.group('{');return ('fraction',top,bottom)
        if token in self.functions:
            base=None
            if token=='log' and self.peek()=='_':self.take('_');base=self.atom()
            self.take('(');arguments=[self.relation()]
            while self.peek() in (',',';'):self.take();arguments.append(self.relation())
            self.take(')')
            if token not in ('max','min') and len(arguments)!=1:raise ValueError('Unexpected function arguments')
            return ('function',token,tuple(arguments),base)
        if re.fullmatch(self.number+r'(?:[eE][+−-]?[0-9]{1,3})?',token):
            for count in range(min(5,len(self.tokens)-self.i),0,-1):
                symbol=''.join(self.tokens[self.i:self.i+count])
                if self.units.fullmatch(symbol):self.i+=count;return ('quantity',token,symbol)
            return ('number',token)
        if token in self.greek:return ('greek',token)
        if token=='∞':return ('infinity',)
        if re.fullmatch('[A-Za-z]',token):return ('variable',token.upper())
        if token in FRACTIONS:
            top,bottom=FRACTIONS[token];return ('fraction',('number',str(top)),('number',str(bottom)))
        raise ValueError('Unsupported math token')
