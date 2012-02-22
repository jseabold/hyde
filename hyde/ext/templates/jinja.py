# -*- coding: utf-8 -*-
"""
Jinja template utilties
"""

from datetime import datetime, date
import os
import re
import itertools
from urllib import quote, unquote
import cStringIO

from hyde.fs import File, Folder
from hyde.model import Expando
from hyde.template import HtmlWrap, Template
from hyde.util import getLoggerWithNullHandler
from operator import attrgetter

from jinja2 import contextfunction, Environment
from jinja2 import FileSystemLoader, FileSystemBytecodeCache
from jinja2 import contextfilter, environmentfilter, Markup, Undefined, nodes
from jinja2.ext import Extension
from jinja2.exceptions import TemplateError

try: # dependencies for ipython filter
    from IPython import Config, InteractiveShell
    from IPython.core.profiledir import ProfileDir
    from IPython.utils import io
    import tempfile
    import sys
    import ast
except ImportError:
    has_ipython = False

try:
    from pygments.lexer import Lexer, do_insertions
    from pygments.lexers.agile import (PythonConsoleLexer, PythonLexer,
                                   PythonTracebackLexer)
    from pygments.token import Comment, Generic
except:
    pass # this is checked again in syntax, so don't worry about it

logger = getLoggerWithNullHandler('hyde.engine.Jinja2')

class SilentUndefined(Undefined):
    """
    A redefinition of undefined that eats errors.
    """
    def __getattr__(self, name):
        return self

    __getitem__ = __getattr__

    def __call__(self, *args, **kwargs):
        return self

@contextfunction
def media_url(context, path, safe=None):
    """
    Returns the media url given a partial path.
    """
    return context['site'].media_url(path, safe)

@contextfunction
def content_url(context, path, safe=None):
    """
    Returns the content url given a partial path.
    """
    return context['site'].content_url(path, safe)

@contextfunction
def full_url(context, path, safe=None):
    """
    Returns the full url given a partial path.
    """
    return context['site'].full_url(path, safe)

@contextfilter
def urlencode(ctx, url, safe=None):
    if safe is not None:
        return quote(url.encode('utf8'), safe)
    else:
        return quote(url.encode('utf8'))

@contextfilter
def urldecode(ctx, url):
    return unquote(url).decode('utf8')

@contextfilter
def date_format(ctx, dt, fmt=None):
    if not dt:
        dt = datetime.now()
    if not isinstance(dt, datetime) or \
        not isinstance(dt, date):
        logger.error("Date format called on a non date object")
        return dt

    format = fmt or "%a, %d %b %Y"
    if not fmt:
        global_format = ctx.resolve('dateformat')
        if not isinstance(global_format, Undefined):
            format = global_format
    return dt.strftime(format)


def islice(iterable, start=0, stop=3, step=1):
    return itertools.islice(iterable, start, stop, step)

def top(iterable, count=3):
    return islice(iterable, stop=count)

def xmldatetime(dt):
    if not dt:
        dt = datetime.now()
    zprefix = "Z"
    tz = dt.strftime("%z")
    if tz:
        zprefix = tz[:3] + ":" + tz[3:]
    return dt.strftime("%Y-%m-%dT%H:%M:%S") + zprefix

@environmentfilter
def asciidoc(env, value):
    """
    (simple) Asciidoc filter
    """
    try:
        from asciidocapi import AsciiDocAPI
    except ImportError:
        print u"Requires AsciiDoc library to use AsciiDoc tag."
        raise

    import StringIO
    output = value

    asciidoc = AsciiDocAPI()
    asciidoc.options('--no-header-footer')
    result = StringIO.StringIO()
    asciidoc.execute(StringIO.StringIO(output.encode('utf-8')), result, backend='html4')
    return unicode(result.getvalue(), "utf-8")

@environmentfilter
def markdown(env, value):
    """
    Markdown filter with support for extensions.
    """
    try:
        import markdown as md
    except ImportError:
        logger.error(u"Cannot load the markdown library.")
        raise TemplateError(u"Cannot load the markdown library")
    output = value
    d = {}
    if hasattr(env.config, 'markdown'):
        d['extensions'] = getattr(env.config.markdown, 'extensions', [])
        d['extension_configs'] = getattr(env.config.markdown,
                                        'extension_configs',
                                        Expando({})).to_dict()
        if hasattr(env.config.markdown, 'output_format'):
            d['output_format'] = env.config.markdown.output_format
    marked = md.Markdown(**d)

    return marked.convert(output)

@environmentfilter
def restructuredtext(env, value):
    """
    RestructuredText filter
    """
    try:
        from docutils.core import publish_parts
    except ImportError:
        logger.error(u"Cannot load the docutils library.")
        raise TemplateError(u"Cannot load the docutils library.")

    highlight_source = False
    if hasattr(env.config, 'restructuredtext'):
        highlight_source = getattr(env.config.restructuredtext, 'highlight_source', False)

    if highlight_source:
        import hyde.lib.pygments.rst_directive

    parts = publish_parts(source=value, writer_name="html")
    return parts['html_body']

class IPythonConsoleLexer(Lexer):
    """
    For IPython console output or doctests, such as:

    .. sourcecode:: ipython

      In [1]: a = 'foo'

      In [2]: a
      Out[2]: 'foo'

      In [3]: print a
      foo

      In [4]: 1 / 0

    Notes:

      - Tracebacks are not currently supported.

      - It assumes the default IPython prompts, not customized ones.
    """

    name = 'IPython console session'
    aliases = ['ipython']
    mimetypes = ['text/x-ipython-console']
    input_prompt = re.compile("(In \[[0-9]+\]: )|(   \.\.\.+:)")
    output_prompt = re.compile("(Out\[[0-9]+\]: )|(   \.\.\.+:)")
    continue_prompt = re.compile("   \.\.\.+:")
    tb_start = re.compile("\-+")

    def get_tokens_unprocessed(self, text):
        pylexer = PythonLexer(**self.options)
        tblexer = PythonTracebackLexer(**self.options)

        curcode = ''
        insertions = []
        line_re = re.compile('.*?\n')
        for match in line_re.finditer(text):
            line = match.group()
            input_prompt = self.input_prompt.match(line)
            continue_prompt = self.continue_prompt.match(line.rstrip())
            output_prompt = self.output_prompt.match(line)
            if line.startswith("#"):
                insertions.append((len(curcode),
                                   [(0, Comment, line)]))
            elif input_prompt is not None:
                insertions.append((len(curcode),
                                   [(0, Generic.Prompt, input_prompt.group())]))
                curcode += line[input_prompt.end():]
            elif continue_prompt is not None:
                insertions.append((len(curcode),
                                   [(0, Generic.Prompt, continue_prompt.group())]))
                curcode += line[continue_prompt.end():]
            elif output_prompt is not None:
                # Use the 'error' token for output.  We should probably make
                # our own token, but error is typicaly in a bright color like
                # red, so it works fine for our output prompts.
                insertions.append((len(curcode),
                                   [(0, Generic.Error, output_prompt.group())]))
                curcode += line[output_prompt.end():]
            else:
                if curcode:
                    for item in do_insertions(insertions,
                                              pylexer.get_tokens_unprocessed(curcode)):
                        yield item
                        curcode = ''
                        insertions = []
                yield match.start(), Generic.Output, line
        if curcode:
            for item in do_insertions(insertions,
                                      pylexer.get_tokens_unprocessed(curcode)):
                yield item

@environmentfilter
def syntax(env, value, lexer=None, filename=None):
    """
    Processes the contained block using `pygments`
    """
    try:
        import pygments
        from pygments import lexers
        from pygments import formatters
    except ImportError:
        logger.error(u"pygments library is required to"
                        " use syntax highlighting tags.")
        raise TemplateError("Cannot load pygments")

    if lexer == 'ipython': # not sure how else to register the syntax
        pyg = IPythonConsoleLexer()
    else:
        pyg = (lexers.get_lexer_by_name(lexer)
                if lexer else
                    lexers.guess_lexer(value))
    settings = {}
    if hasattr(env.config, 'syntax'):
        settings = getattr(env.config.syntax,
                            'options',
                            Expando({})).to_dict()

    formatter = formatters.HtmlFormatter(**settings)
    code = pygments.highlight(value, pyg, formatter)
    code = code.replace('\n\n', '\n&nbsp;\n').replace('\n', '<br />')
    caption = filename if filename else pyg.name
    if lexer == 'ipython': # deal with images in ipython syntax, also put them
                           # in facebox, this should be done in javascript
                           # at the DOM level...
        pat = "(&lt;img src=&quot;)(.+)(&quot; /&gt;)"
        code = re.sub(pat, "<a href = \"\\2\" rel=\"facebox\"><img src=\"\\2\" /></a>", code)
    if hasattr(env.config, 'syntax'):
        if not getattr(env.config.syntax, 'use_figure', True):
            return Markup(code)
    return Markup(
            '<div class="codebox"><figure class="code">%s<figcaption>%s</figcaption></figure></div>\n\n'
                        % (code, caption))

class EmbeddedJinjaShell(object):
    """An embedded IPython instance to run inside Jinja"""

    def __init__(self):

        self.cout = cStringIO.StringIO()

        # for tokenizing blocks
        self.COMMENT, self.INPUT, self.OUTPUT = range(3)


        # Create config object for IPython
        config = Config()
        config.Global.display_banner = False
        config.Global.exec_lines = ['import numpy as np',
                                    'from pylab import *'
                                    ]
        config.InteractiveShell.autocall = False
        config.InteractiveShell.autoindent = False
        config.InteractiveShell.colors = 'NoColor'

        # create a profile so instance history isn't saved
        tmp_profile_dir = tempfile.mkdtemp(prefix='profile_')
        profname = 'auto_profile_jinja_build'
        pdir = os.path.join(tmp_profile_dir,profname)
        profile = ProfileDir.create_profile_dir(pdir)

        # Create and initialize ipython, but don't start its mainloop
        IP = InteractiveShell.instance(config=config, profile_dir=profile)
        # io.stdout redirect must be done *after* instantiating InteractiveShell
        io.stdout = self.cout
        io.stderr = self.cout

        # For debugging, so we can see normal output, use this:
        #from IPython.utils.io import Tee
        #io.stdout = Tee(self.cout, channel='stdout') # dbg
        #io.stderr = Tee(self.cout, channel='stderr') # dbg

        # Store a few parts of IPython we'll need.
        self.IP = IP
        self.user_ns = self.IP.user_ns
        self.user_global_ns = self.IP.user_global_ns

        self.input = ''
        self.output = ''

        self.is_verbatim = False
        self.is_doctest = False
        self.is_suppress = False

        # on the first call to the savefig decorator, we'll import
        # pyplot as plt so we can make a call to the plt.gcf().savefig
        self._pyplot_imported = False

    def clear_cout(self):
        self.cout.seek(0)
        self.cout.truncate(0)

    def block_parser(self, part, rgxin, rgxout, fmtin, fmtout):
        """
        part is a string of ipython text, comprised of at most one
        input, one ouput, comments, and blank lines.  The block parser
        parses the text into a list of::

          blocks = [ (TOKEN0, data0), (TOKEN1, data1), ...]

        where TOKEN is one of [COMMENT | INPUT | OUTPUT ] and
        data is, depending on the type of token::

          COMMENT : the comment string

          INPUT: the (DECORATOR, INPUT_LINE, REST) where
             DECORATOR: the input decorator (or None)
             INPUT_LINE: the input as string (possibly multi-line)
             REST : any stdout generated by the input line (not OUTPUT)


          OUTPUT: the output string, possibly multi-line
        """

        block = []
        lines = part.split('\n')
        N = len(lines)
        i = 0
        decorator = None
        while 1:

            if i==N:
                # nothing left to parse -- the last line
                break

            line = lines[i]
            i += 1
            line_stripped = line.strip()
            if line_stripped.startswith('#'):
                block.append((self.COMMENT, line))
                continue

            if line_stripped.startswith('@'):
                # we're assuming at most one decorator -- may need to
                # rethink
                decorator = line_stripped
                continue

            # does this look like an input line?
            matchin = rgxin.match(line)
            if matchin:
                lineno, inputline = int(matchin.group(1)), matchin.group(2)

                # the ....: continuation string
                continuation = '   %s:'% ''.join(['.']*(len(str(lineno))+2))
                Nc = len(continuation)
                # input lines can continue on for more than one line, if
                # we have a '\' line continuation char or a function call
                # echo line 'print'.  The input line can only be
                # terminated by the end of the block or an output line, so
                # we parse out the rest of the input line if it is
                # multiline as well as any echo text

                rest = []
                while i<N:

                    # look ahead; if the next line is blank, or a comment, or
                    # an output line, we're done

                    nextline = lines[i]
                    matchout = rgxout.match(nextline)
                    #print "nextline=%s, continuation=%s, starts=%s"%(nextline, continuation, nextline.startswith(continuation))
                    if matchout or nextline.startswith('#'):
                        break
                    elif nextline.startswith(continuation):
                        inputline += '\n' + nextline[Nc:]
                    else:
                        rest.append(nextline)
                    i+= 1

                block.append((self.INPUT, (decorator, inputline, '\n'.join(rest))))
                continue

            # if it looks like an output line grab all the text to the end
            # of the block
            matchout = rgxout.match(line)
            if matchout:
                lineno, output = int(matchout.group(1)), matchout.group(2)
                if i<N-1:
                    output = '\n'.join([output] + lines[i:])

                block.append((self.OUTPUT, output))
                break

        return block

    def process_input_line(self, line, store_history=True):
        """process the input, capturing stdout"""
        #print "input='%s'"%self.input
        stdout = sys.stdout
        splitter = self.IP.input_splitter
        try:
            sys.stdout = self.cout
            splitter.push(line)
            more = splitter.push_accepts_more()
            if not more:
                source_raw = splitter.source_raw_reset()[1]
                self.IP.run_cell(source_raw, store_history=store_history)
        finally:
            sys.stdout = stdout

    def process_image(self, decorator):
        """
        # build out an image directive like
        # ![Alt text](somefile.png)
        #    :width 4in
        #
        # from an input like
        # savefig somefile.png width=4in
        """
        savefig_dir = self.savefig_dir
        saveargs = decorator.split(' ')
        filename = saveargs[1]
        # insert relative path to image file in source
        outfile = '/' + os.path.join(savefig_dir,filename)

        imagerows = ['<img src="%s" />'%outfile]

        for kwarg in saveargs[2:]:
            arg, val = kwarg.split('=')
            arg = arg.strip()
            val = val.strip()
            imagerows.append('   :%s: %s'%(arg, val))

        image_file = os.path.basename(outfile) # only return file name
        image_directive = '\n'.join(imagerows)
        return image_file, image_directive


    # Callbacks for each type of token
    def process_input(self, data, input_prompt, lineno):
        """Process data block for INPUT token."""
        decorator, input, rest = data
        image_file = None
        image_directive = None
        #print 'INPUT:', data  # dbg
        is_verbatim = decorator=='@verbatim' or self.is_verbatim
        is_doctest = decorator=='@doctest' or self.is_doctest
        is_suppress = decorator=='@suppress' or self.is_suppress
        is_savefig = decorator is not None and \
                     decorator.startswith('@savefig')

        input_lines = input.split('\n')
        if len(input_lines) > 1:
            if input_lines[-1] != "":
                input_lines.append('') # make sure there's blank line
                                       # so splitter buffer gets reset

        continuation = '   %s:'%''.join(['.']*(len(str(lineno))+2))
        Nc = len(continuation)

        if is_savefig:
            image_file, image_directive = self.process_image(decorator)

        ret = []
        is_semicolon = False
        store_history = True

        for i, line in enumerate(input_lines):
            if line.endswith(';'):
                is_semicolon = True
            if is_suppress:
                store_history = False

            if i==0:
                # process the first input line
                if is_verbatim:
                    self.process_input_line('')
                    self.IP.execution_count += 1 # increment it anyway
                else:
                    # only submit the line in non-verbatim mode
                    self.process_input_line(line, store_history=store_history)
                formatted_line = '%s %s'%(input_prompt, line)
            else:
                # process a continuation line
                if not is_verbatim:
                    self.process_input_line(line, store_history=store_history)

                formatted_line = '%s%s'%(continuation, line)
            if not is_suppress:
                ret.append(formatted_line)

        if not is_suppress and len(rest.strip()) and is_verbatim:
            # the "rest" is the standard output of the
            # input, which needs to be added in
            # verbatim mode
            ret.append(rest)

        self.cout.seek(0)
        output = self.cout.read()
        if not is_suppress and not is_semicolon:
            ret.append(output)
        elif is_semicolon: # to get the spacing right
            ret.append('')

        self.cout.truncate(0)
        return (ret, input_lines, output, is_doctest, image_file,
                    image_directive)
        #print 'OUTPUT', output  # dbg

    def process_output(self, data, output_prompt,
                       input_lines, output, is_doctest, image_file):
        """Process data block for OUTPUT token."""
        if is_doctest:
            submitted = data.strip()
            found = output
            if found is not None:
                found = found.strip()
                ind = found.find(output_prompt)
                if ind<0:
                    e='output prompt="%s" does not match out line=%s' % \
                       (output_prompt, found)
                    raise RuntimeError(e)
                found = found[len(output_prompt):].strip()

                if found!=submitted:
                    e = ('doctest failure for input_lines="%s" with '
                         'found_output="%s" and submitted output="%s"' %
                         (input_lines, found, submitted) )
                    raise RuntimeError(e)
                #print 'doctest PASSED for input_lines="%s" with found_output="%s" and submitted output="%s"'%(input_lines, found, submitted)

    def process_comment(self, data):
        """Process data fPblock for COMMENT token."""
        if not self.is_suppress:
            return [data]

    def save_image(self, image_file):
        """
        Saves the image file to disk.
        """
        image_dir = os.path.join(self.source_dir,
                                 self.content_dir,
                                 self.savefig_dir)
        image_file = os.path.join(image_dir, image_file)
        print "Saving image to %s" % image_file
        self.ensure_pyplot()
        command = ('plt.gcf().savefig("%s", bbox_inches="tight", '
                   'dpi=100)' % image_file)
        #print 'SAVEFIG', command  # dbg
        self.process_input_line('bookmark ipy_thisdir', store_history=False)
        self.process_input_line('cd -b ipy_savedir', store_history=False)
        self.process_input_line(command, store_history=False)
        self.process_input_line('cd -b ipy_thisdir', store_history=False)
        self.process_input_line('bookmark -d ipy_thisdir', store_history=False)
        self.clear_cout()


    def process_block(self, block):
        """
        process block from the block_parser and return a list of processed lines
        """
        ret = []
        output = None
        input_lines = None
        lineno = self.IP.execution_count

        input_prompt = self.promptin%lineno
        output_prompt = self.promptout%lineno
        image_file = None
        image_directive = None

        for token, data in block:
            if token==self.COMMENT:
                out_data = self.process_comment(data)
            elif token==self.INPUT:
                (out_data, input_lines, output, is_doctest, image_file,
                    image_directive) = \
                          self.process_input(data, input_prompt, lineno)
            elif token==self.OUTPUT:
                out_data = \
                    self.process_output(data, output_prompt,
                                        input_lines, output, is_doctest,
                                        image_file)
            if out_data:
                ret.extend(out_data)

        # save the image files
        if image_file is not None:
            self.save_image(image_file)

        return ret, image_directive

    def ensure_pyplot(self):
        if self._pyplot_imported:
            return
        self.process_input_line('import matplotlib.pyplot as plt',
                                store_history=False)

    def process_pure_python(self, content):
        """
        content is a list of strings. it is unedited directive conent

        This runs it line by line in the InteractiveShell, prepends
        prompts as needed capturing stderr and stdout, then returns
        the content as a list as if it were ipython code
        """
        output = []
        savefig = False # keep up with this to clear figure
        multiline = False # to handle line continuation
        fmtin = self.promptin

        for lineno, line in enumerate(content):

            line_stripped = line.strip()

            if not len(line):
                output.append(line) # preserve empty lines in output
                continue

            # handle decorators
            if line_stripped.startswith('@'):
                output.extend([line])
                if 'savefig' in line:
                    savefig = True # and need to clear figure
                continue

            # handle comments
            if line_stripped.startswith('#'):
                output.extend([line])
                continue

            # deal with multilines
            if not multiline: # not currently on a multiline

                if line_stripped.endswith('\\'): # now we are
                    multiline = True
                    cont_len = len(str(lineno)) + 2
                    line_to_process = line.strip('\\')
                    output.extend([u"%s %s" % (fmtin%lineno,line)])
                    continue
                else: # no we're still not
                    line_to_process = line.strip('\\')
            else: # we are currently on a multiline
                line_to_process += line.strip('\\')
                if line_stripped.endswith('\\'): # and we still are
                    continuation = '.' * cont_len
                    output.extend([(u'   %s: '+line_stripped) % continuation])
                    continue
                # else go ahead and run this multiline then carry on

            # get output of line
            self.process_input_line(unicode(line_to_process.strip()),
                                    store_history=False)
            out_line = self.cout.getvalue()
            self.clear_cout()

            # clear current figure if plotted
            if savefig:
                self.ensure_pyplot()
                self.process_input_line('plt.clf()', store_history=False)
                self.clear_cout()
                savefig = False

            # line numbers don't actually matter, they're replaced later
            if not multiline:
                in_line = u"%s %s" % (fmtin%lineno,line)

                output.extend([in_line])
            else:
                output.extend([(u'   %s: '+line_stripped) % continuation])
                multiline = False
            if len(out_line):
                output.extend([out_line])
            output.extend([u''])

        return output

    def process_pure_python2(self, content):
        """
        content is a list of strings. it is unedited directive conent

        This runs it line by line in the InteractiveShell, prepends
        prompts as needed capturing stderr and stdout, then returns
        the content as a list as if it were ipython code
        """
        output = []
        savefig = False # keep up with this to clear figure
        multiline = False # to handle line continuation
        multiline_start = None
        fmtin = self.promptin

        ct = 0

        for lineno, line in enumerate(content):

            line_stripped = line.strip()
            if not len(line):
                output.append(line)
                continue

            # handle decorators
            if line_stripped.startswith('@'):
                output.extend([line])
                if 'savefig' in line:
                    savefig = True # and need to clear figure
                continue

            # handle comments
            if line_stripped.startswith('#'):
                output.extend([line])
                continue

            continuation  = u'   %s:'% ''.join(['.']*(len(str(ct))+2))
            if not multiline:
                modified = u"%s %s" % (fmtin % ct, line_stripped)
                output.append(modified)
                ct += 1
                try:
                    ast.parse(line_stripped)
                    output.append(u'')
                except Exception:
                    multiline = True
                    multiline_start = lineno
                    if line_stripped.startswith('def '):
                        is_function = True
            else:
                modified = u'%s %s' % (continuation, line)
                output.append(modified)
                try:
                    mod = ast.parse(
                            '\n'.join(content[multiline_start:lineno+1]))
                    if isinstance(mod.body[0], ast.FunctionDef):
                        # check to see if we have the whole function
                        for element in mod.body[0].body:
                            if isinstance(element, ast.Return):
                                multiline = False
                    else:
                        output.append(u'')
                        multiline = False
                except Exception:
                    pass

            if savefig: # clear figure if plotted
                self.ensure_pyplot()
                self.process_input_line('plt.clf()', store_history=False)
                self.clear_cout()
                savefig = False

        return output

@environmentfilter
def ipython(env, value):
    """
    IPython filter.
    """
    shell = EmbeddedJinjaShell()
    old_exec_count = [i for i in os.listdir(tempfile.tempdir)
                      if i.startswith('exec_count')]
    if old_exec_count:
        fname = os.path.join(tempfile.tempdir, old_exec_count[0])
        shell.IP.history_manager.reset()
        shell.IP.execution_count -= (int(open(fname).read()) - 1)
        os.remove(fname)

    # setup the shell
    import re

    if hasattr(env.config, 'ipythontext'):
        shell.rgxin = getattr(env.config.ipythontext, 'rgxin' ,
                        re.compile('In \[(\d+)\]:\s?(.*)\s*') )
        shell.rgxout = getattr(env.config.ipythontext, 'rgxout',
                        re.compile('Out\[(\d+)\]:\s?(.*)\s*'))
        shell.promptin = getattr(env.config.ipythontext, 'promptin', 'In [%d]:')
        shell.promptout = getattr(env.config.ipythontext, 'promptout', 'Out[%d]:')
        shell.savefig_dir = getattr(env.config.ipythontext, 'savefig_dir',
                            env.config.media_url)
    else:
        shell.rgxin = re.compile('In \[(\d+)\]:\s?(.*)\s*')
        shell.rgxout = re.compile('Out\[(\d+)\]:\s?(.*)\s*')
        shell.promptin = 'In [%d]:'
        shell.promptout = 'Out[%d]:'
        shell.savefig_dir = os.path.join(env.config.media_url[1:],
                                         'img/')
        shell.content_dir = env.config.content_root

    shell.source_dir = env.config.sitepath.path


    rgxin, rgxout = shell.rgxin, shell.rgxout
    promptin, promptout = shell.promptin, shell.promptout


    shell.process_input_line('bookmark ipy_savedir %s' % shell.savefig_dir,
                                      store_history=False)
    shell.clear_cout()

    shell.is_suppress = False
    shell.is_doctest = False
    shell.is_verbatim = False

    # we need a list
    content = value.strip().split('\n')

    #NOTE: this assumes we've gotten pure python, but doesn't need to be
    # run if it's an IPython session that's copy pasted
    content = shell.process_pure_python2(content)

    parts = '\n'.join(content).split('\n\n')

    # this writes rst because I'm too lazy to update it
    lines = []
    figures = []
    for part in parts:
        block = shell.block_parser(part, rgxin, rgxout, promptin, promptout)

        if len(block):
            rows, figure = shell.process_block(block)
            for row in rows:
                lines.extend(['%s' % line
                              for line in re.split('[\n]+', row)])
            if figure is not None:
                figures.append(figure)

    for figure in figures:
        lines.append('')
        lines.extend(figure.split('\n'))
        lines.append('')

    text = '\n'.join(lines)

    # save execution count, don't know why hyde runs this all twice
    # this is going to cause problems for mutable objects re-used in state
    # if they change
    fd, fname = tempfile.mkstemp(prefix="exec_count", text=True)
    with open(fname, 'w') as fout:
        fout.write(str(shell.IP.execution_count))

    return syntax(env, text, 'ipython')

class Spaceless(Extension):
    """
    Emulates the django spaceless template tag.
    """

    tags = set(['spaceless'])

    def parse(self, parser):
        """
        Parses the statements and calls back to strip spaces.
        """
        lineno = parser.stream.next().lineno
        body = parser.parse_statements(['name:endspaceless'],
                drop_needle=True)
        return nodes.CallBlock(
                    self.call_method('_render_spaceless'),
                    [], [], body).set_lineno(lineno)

    def _render_spaceless(self, caller=None):
        """
        Strip the spaces between tags using the regular expression
        from django. Stolen from `django.util.html` Returns the given HTML
        with spaces between tags removed.
        """
        if not caller:
            return ''
        return re.sub(r'>\s+<', '><', unicode(caller().strip()))

class Asciidoc(Extension):
    """
    A wrapper around the asciidoc filter for syntactic sugar.
    """
    tags = set(['asciidoc'])

    def parse(self, parser):
        """
        Parses the statements and defers to the callback for asciidoc processing.
        """
        lineno = parser.stream.next().lineno
        body = parser.parse_statements(['name:endasciidoc'], drop_needle=True)

        return nodes.CallBlock(
                    self.call_method('_render_asciidoc'),
                        [], [], body).set_lineno(lineno)

    def _render_asciidoc(self, caller=None):
        """
        Calls the asciidoc filter to transform the output.
        """
        if not caller:
            return ''
        output = caller().strip()
        return asciidoc(self.environment, output)

class Markdown(Extension):
    """
    A wrapper around the markdown filter for syntactic sugar.
    """
    tags = set(['markdown'])

    def parse(self, parser):
        """
        Parses the statements and defers to the callback for markdown processing.
        """
        lineno = parser.stream.next().lineno
        body = parser.parse_statements(['name:endmarkdown'], drop_needle=True)

        return nodes.CallBlock(
                    self.call_method('_render_markdown'),
                        [], [], body).set_lineno(lineno)

    def _render_markdown(self, caller=None):
        """
        Calls the markdown filter to transform the output.
        """
        if not caller:
            return ''
        output = caller().strip()
        return markdown(self.environment, output)

class restructuredText(Extension):
    """
    A wrapper around the restructuredtext filter for syntactic sugar
    """
    tags = set(['restructuredtext'])

    def parse(self, parser):
        """
        Simply extract our content
        """
        lineno = parser.stream.next().lineno
        body = parser.parse_statements(['name:endrestructuredtext'], drop_needle=True)

        return nodes.CallBlock(self.call_method('_render_rst'), [],  [], body
                              ).set_lineno(lineno)

    def _render_rst(self, caller=None):
        """
        call our restructuredtext filter
        """
        if not caller:
            return ''
        output = caller().strip()
        return restructuredtext(self.environment, output)

class YamlVar(Extension):
    """
    An extension that converts the content between the tags
    into an yaml object and sets the value in the given
    variable.
    """

    tags = set(['yaml'])

    def parse(self, parser):
        """
        Parses the contained data and defers to the callback to load it as
        yaml.
        """
        lineno = parser.stream.next().lineno
        var = parser.stream.expect('name').value
        body = parser.parse_statements(['name:endyaml'], drop_needle=True)
        return [
                nodes.Assign(
                    nodes.Name(var, 'store'),
                    nodes.Const({})
                    ).set_lineno(lineno),
                nodes.CallBlock(
                    self.call_method('_set_yaml',
                            args=[nodes.Name(var, 'load')]),
                            [], [], body).set_lineno(lineno)
                ]


    def _set_yaml(self, var, caller=None):
        """
        Loads the yaml data into the specified variable.
        """
        if not caller:
            return ''
        try:
            import yaml
        except ImportError:
            return ''

        out = caller().strip()
        var.update(yaml.load(out))
        return ''

def parse_kwargs(parser):
    """
    Parses keyword arguments in tags.
    """
    name = parser.stream.expect('name').value
    parser.stream.expect('assign')
    if parser.stream.current.test('string'):
        value = parser.parse_expression()
    else:
        value = nodes.Const(parser.stream.next().value)
    return (name, value)

class Syntax(Extension):
    """
    A wrapper around the syntax filter for syntactic sugar.
    """

    tags = set(['syntax'])


    def parse(self, parser):
        """
        Parses the statements and defers to the callback for pygments processing.
        """
        lineno = parser.stream.next().lineno
        lex = nodes.Const(None)
        filename = nodes.Const(None)

        if not parser.stream.current.test('block_end'):
            if parser.stream.look().test('assign'):
                name = value = value1 = None
                (name, value) = parse_kwargs(parser)
                if parser.stream.skip_if('comma'):
                    (_, value1) = parse_kwargs(parser)

                (lex, filename) = (value, value1) \
                                        if name == 'lex' \
                                            else (value1, value)
            else:
                lex = nodes.Const(parser.stream.next().value)
                if parser.stream.skip_if('comma'):
                    filename = parser.parse_expression()

        body = parser.parse_statements(['name:endsyntax'], drop_needle=True)
        return nodes.CallBlock(
                    self.call_method('_render_syntax',
                        args=[lex, filename]),
                        [], [], body).set_lineno(lineno)


    def _render_syntax(self, lex, filename, caller=None):
        """
        Calls the syntax filter to transform the output.
        """
        if not caller:
            return ''
        output = caller().strip()
        return syntax(self.environment, output, lex, filename)

class IncludeText(Extension):
    """
    Automatically runs `markdown` and `typogrify` on included
    files.
    """

    tags = set(['includetext'])

    def parse(self, parser):
        """
        Delegates all the parsing to the native include node.
        """
        node = parser.parse_include()
        return nodes.CallBlock(
                    self.call_method('_render_include_text'),
                        [], [], [node]).set_lineno(node.lineno)

    def _render_include_text(self, caller=None):
        """
        Runs markdown and if available, typogrigy on the
        content returned by the include node.
        """
        if not caller:
            return ''
        output = caller().strip()
        output = markdown(self.environment, output)
        if 'typogrify' in self.environment.filters:
            typo = self.environment.filters['typogrify']
            output = typo(output)
        return output

MARKINGS = '_markings_'

class Reference(Extension):
    """
    Marks a block in a template such that its available for use
    when referenced using a `refer` tag.
    """

    tags = set(['mark', 'reference'])

    def parse(self, parser):
        """
        Parse the variable name that the content must be assigned to.
        """
        token = parser.stream.next()
        lineno = token.lineno
        tag = token.value
        name = parser.stream.next().value
        body = parser.parse_statements(['name:end%s' % tag], drop_needle=True)
        return nodes.CallBlock(
                    self.call_method('_render_output',
                        args=[nodes.Name(MARKINGS, 'load'), nodes.Const(name)]),
                        [], [], body).set_lineno(lineno)


    def _render_output(self, markings, name, caller=None):
        """
        Assigns the result of the contents to the markings variable.
        """
        if not caller:
            return ''
        out = caller()
        if isinstance(markings, dict):
            markings[name] = out
        return out

class Refer(Extension):
    """
    Imports content blocks specified in the referred template as
    variables in a given namespace.
    """
    tags = set(['refer'])

    def parse(self, parser):
        """
        Parse the referred template and the namespace.
        """
        token = parser.stream.next()
        lineno = token.lineno
        parser.stream.expect('name:to')
        template = parser.parse_expression()
        parser.stream.expect('name:as')
        namespace = parser.stream.next().value
        includeNode = nodes.Include(lineno=lineno)
        includeNode.with_context = True
        includeNode.ignore_missing = False
        includeNode.template = template

        temp = parser.free_identifier(lineno)

        return [
                nodes.Assign(
                    nodes.Name(temp.name, 'store'),
                    nodes.Name(MARKINGS, 'load')
                ).set_lineno(lineno),
                nodes.Assign(
                    nodes.Name(MARKINGS, 'store'),
                    nodes.Const({})).set_lineno(lineno),
                nodes.Assign(
                    nodes.Name(namespace, 'store'),
                    nodes.Const({})).set_lineno(lineno),
                nodes.CallBlock(
                    self.call_method('_push_resource',
                            args=[
                                nodes.Name(namespace, 'load'),
                                nodes.Name('site', 'load'),
                                nodes.Name('resource', 'load'),
                                template]),
                            [], [], []).set_lineno(lineno),
                nodes.Assign(
                    nodes.Name('resource', 'store'),
                    nodes.Getitem(nodes.Name(namespace, 'load'),
                        nodes.Const('resource'), 'load')
                    ).set_lineno(lineno),
                nodes.CallBlock(
                    self.call_method('_assign_reference',
                            args=[
                                nodes.Name(MARKINGS, 'load'),
                                nodes.Name(namespace, 'load')]),
                            [], [], [includeNode]).set_lineno(lineno),
                nodes.Assign(nodes.Name('resource', 'store'),
                            nodes.Getitem(nodes.Name(namespace, 'load'),
                            nodes.Const('parent_resource'), 'load')
                    ).set_lineno(lineno),
                    nodes.Assign(
                        nodes.Name(MARKINGS, 'store'),
                        nodes.Name(temp.name, 'load')
                    ).set_lineno(lineno),
        ]

    def _push_resource(self, namespace, site, resource, template, caller):
        """
        Saves the current references in a stack.
        """
        namespace['parent_resource'] = resource
        if not hasattr(resource, 'depends'):
            resource.depends = []
        if not template in resource.depends:
            resource.depends.append(template)
        namespace['resource'] = site.content.resource_from_relative_path(
                                    template)
        return ''

    def _assign_reference(self, markings, namespace, caller):
        """
        Assign the processed variables into the
        given namespace.
        """

        out = caller()
        for key, value in markings.items():
            namespace[key] = value
        namespace['html'] = HtmlWrap(out)
        return ''


class HydeLoader(FileSystemLoader):
    """
    A wrapper around the file system loader that performs
    hyde specific tweaks.
    """

    def __init__(self, sitepath, site, preprocessor=None):
        config = site.config if hasattr(site, 'config') else None
        if config:
            super(HydeLoader, self).__init__([
                            unicode(config.content_root_path),
                            unicode(config.layout_root_path),
                        ])
        else:
            super(HydeLoader, self).__init__(unicode(sitepath))

        self.site = site
        self.preprocessor = preprocessor

    def get_source(self, environment, template):
        """
        Calls the plugins to preprocess prior to returning the source.
        """
        template = template.strip()
        # Fixed so that jinja2 loader does not have issues with
        # seprator in windows
        #
        template = template.replace(os.sep, '/')
        logger.debug("Loading template [%s] and preprocessing" % template)
        (contents,
            filename,
                date) = super(HydeLoader, self).get_source(
                                        environment, template)
        if self.preprocessor:
            resource = self.site.content.resource_from_relative_path(template)
            if resource:
                contents = self.preprocessor(resource, contents) or contents
        return (contents, filename, date)


# pylint: disable-msg=W0104,E0602,W0613,R0201
class Jinja2Template(Template):
    """
    The Jinja2 Template implementation
    """

    def __init__(self, sitepath):
        super(Jinja2Template, self).__init__(sitepath)

    def configure(self, site, engine=None):
        """
        Uses the site object to initialize the jinja environment.
        """
        self.site = site
        self.engine = engine
        self.preprocessor = (engine.preprocessor
                            if hasattr(engine, 'preprocessor') else None)

        self.loader = HydeLoader(self.sitepath, site, self.preprocessor)

        default_extensions = [
                IncludeText,
                Spaceless,
                Asciidoc,
                Markdown,
                restructuredText,
                Syntax,
                Reference,
                Refer,
                YamlVar,
                'jinja2.ext.do',
                'jinja2.ext.loopcontrols',
                'jinja2.ext.with_'
        ]

        defaults = {
            'line_statement_prefix': '$$$',
            'trim_blocks': True,
        }

        settings = dict()
        settings.update(defaults)
        settings['extensions'] = list()
        settings['extensions'].extend(default_extensions)

        conf = {}

        try:
            conf = attrgetter('config.jinja2')(site).to_dict()
        except AttributeError:
            pass

        settings.update(
            dict([(key, conf[key]) for key in defaults if key in conf]))

        extensions = conf.get('extensions', [])
        if isinstance(extensions, list):
            settings['extensions'].extend(extensions)
        else:
            settings['extensions'].append(extensions)

        self.env = Environment(
                    loader=self.loader,
                    undefined=SilentUndefined,
                    line_statement_prefix=settings['line_statement_prefix'],
                    trim_blocks=True,
                    bytecode_cache=FileSystemBytecodeCache(),
                    extensions=settings['extensions'])
        self.env.globals['media_url'] = media_url
        self.env.globals['content_url'] = content_url
        self.env.globals['full_url'] = full_url
        self.env.globals['engine'] = engine
        self.env.globals['deps'] = {}
        self.env.filters['urlencode'] = urlencode
        self.env.filters['urldecode'] = urldecode
        self.env.filters['asciidoc'] = asciidoc
        self.env.filters['markdown'] = markdown
        self.env.filters['restructuredtext'] = restructuredtext
        self.env.filters['syntax'] = syntax
        self.env.filters['date_format'] = date_format
        self.env.filters['xmldatetime'] = xmldatetime
        self.env.filters['islice'] = islice
        self.env.filters['top'] = top
        self.env.filters['ipython'] = ipython

        config = {}
        if hasattr(site, 'config'):
            config = site.config

        self.env.extend(config=config)

        try:
            from typogrify.templatetags import jinja2_filters
        except ImportError:
            jinja2_filters = False

        if jinja2_filters:
            jinja2_filters.register(self.env)

    def clear_caches(self):
        """
        Clear all caches to prepare for regeneration
        """
        if self.env.bytecode_cache:
            self.env.bytecode_cache.clear()


    def get_dependencies(self, path):
        """
        Finds dependencies hierarchically based on the included
        files.
        """
        text = self.env.loader.get_source(self.env, path)[0]
        from jinja2.meta import find_referenced_templates
        try:
            ast = self.env.parse(text)
        except:
            logger.error("Error parsing[%s]" % path)
            raise
        tpls = find_referenced_templates(ast)
        deps = list(self.env.globals['deps'].get('path', []))
        for dep in tpls:
            deps.append(dep)
            if dep:
                deps.extend(self.get_dependencies(dep))
        return list(set(deps))

    @property
    def exception_class(self):
        """
        The exception to throw. Used by plugins.
        """
        return TemplateError

    @property
    def patterns(self):
        """
        The pattern for matching selected template statements.
        """
        return {
           "block_open": '\s*\{\%\s*block\s*([^\s]+)\s*\%\}',
           "block_close": '\s*\{\%\s*endblock\s*([^\s]*)\s*\%\}',
           "include": '\s*\{\%\s*include\s*(?:\'|\")(.+?\.[^.]*)(?:\'|\")\s*\%\}',
           "extends": '\s*\{\%\s*extends\s*(?:\'|\")(.+?\.[^.]*)(?:\'|\")\s*\%\}'
        }

    def get_include_statement(self, path_to_include):
        """
        Returns an include statement for the current template,
        given the path to include.
        """
        return '{%% include \'%s\' %%}' % path_to_include

    def get_extends_statement(self, path_to_extend):
        """
        Returns an extends statement for the current template,
        given the path to extend.
        """
        return '{%% extends \'%s\' %%}' % path_to_extend

    def get_open_tag(self, tag, params):
        """
        Returns an open tag statement.
        """
        return '{%% %s %s %%}' % (tag, params)

    def get_close_tag(self, tag, params):
        """
        Returns an open tag statement.
        """
        return '{%% end%s %%}' % tag

    def get_content_url_statement(self, url):
        """
        Returns the content url statement.
        """
        return '{{ content_url(\'%s\') }}' % url

    def get_media_url_statement(self, url):
        """
        Returns the media url statement.
        """
        return '{{ media_url(\'%s\') }}' % url

    def get_full_url_statement(self, url):
        """
        Returns the full url statement.
        """
        return '{{ full_url(\'%s\') }}' % url

    def render_resource(self, resource, context):
        """
        Renders the given resource using the context
        """
        try:
            template = self.env.get_template(resource.relative_path)
            out = template.render(context)
        except:
            out = ""
            logger.debug(self.env.loader.get_source(
                                self.env, resource.relative_path))
            raise
        return out

    def render(self, text, context):
        """
        Renders the given text using the context
        """
        template = self.env.from_string(text)
        return template.render(context)
