"""ICU's installed CLDR spellout rules, accessed through its stable C API."""
import ctypes as C
import ctypes.util
import re
import threading
import weakref
from functools import lru_cache


@lru_cache(maxsize=1)
def api():
    path=C.util.find_library('icui18n')
    if not path:raise RuntimeError('Multilingual numbers require the system libicui18n library')
    lib=C.CDLL(path);match=re.search(r'\.so\.(\d+)',path);suffix='_'+match[1] if match else ''
    def bind(name,result,args):
        function=getattr(lib,name+suffix);function.restype=result;function.argtypes=args;return function
    u=C.c_uint16;e=C.c_int32;p=C.c_void_p
    return dict(library=lib,version=suffix.lstrip('_'),open=bind('unum_open',p,[e,C.POINTER(u),e,C.c_char_p,p,C.POINTER(e)]),close=bind('unum_close',None,[p]),set=bind('unum_setTextAttribute',None,[p,e,C.POINTER(u),e,C.POINTER(e)]),format=bind('unum_formatInt64',e,[p,C.c_int64,C.POINTER(u),e,p,C.POINTER(e)]))


class Formatter:
    def __init__(self,language,ruleset):
        self.api=api();error=C.c_int32(0)
        self.handle=self.api['open'](5,None,0,language.encode(),None,C.byref(error))
        if not self.handle or error.value>0:raise RuntimeError('Cannot open ICU spellout: '+language)
        self.finalizer=weakref.finalize(self,self.api['close'],self.handle);self.lock=threading.Lock()
        if ruleset:
            encoded=ruleset.encode('utf-16-le');value=(C.c_uint16*(len(encoded)//2)).from_buffer_copy(encoded)
            self.api['set'](self.handle,6,value,len(value),C.byref(error))
            if error.value>0:raise RuntimeError('Unsupported ICU number ruleset: '+ruleset)
    def format(self,value):
        with self.lock:
            error=C.c_int32(0);buffer=(C.c_uint16*2048)()
            length=self.api['format'](self.handle,value,buffer,len(buffer),None,C.byref(error))
            if error.value>0 or not 0<length<len(buffer):raise RuntimeError('ICU number formatting failed')
            return bytes(buffer)[:length*2].decode('utf-16-le').replace('\u00ad','')


@lru_cache(maxsize=32)
def formatter(language,gender=None,case=''):
    rule={'en':'%spellout-cardinal','de':'%spellout-cardinal-'+(gender or 'masculine'),'fr':'%spellout-cardinal-'+(gender or 'masculine'),'es':'%spellout-cardinal-'+(gender or 'masculine'),'ru':'%spellout-cardinal-'+(gender or 'masculine'),'ja':'%spellout-cardinal'}[language]
    if not gender and language in ('de','es'):rule='%spellout-numbering'
    if case:rule+="-"+case
    return Formatter(language,rule)


def integer(value,language,gender=None,case=''):
    if not 0<=value<10**16:raise ValueError('Number outside supported range')
    if language=='ja' and value==0:return 'ゼロ'
    return formatter(language,gender,case).format(value)


def ordinal(value,language,plural=False):
    if language!='ru':raise ValueError('Only the Russian fraction ordinal is used')
    rule='%spellout-ordinal-plural-genitive' if plural else '%spellout-ordinal-feminine'
    return _ordinal_formatter(rule).format(value)


@lru_cache(maxsize=2)
def _ordinal_formatter(rule):return Formatter('ru',rule)
