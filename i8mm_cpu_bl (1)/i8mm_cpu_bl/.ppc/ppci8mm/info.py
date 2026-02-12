from ppcgrader.info_utils import *
from ppcgrader.doc_builder import *
from ppcgrader.reporter import json_to_output

code = "i8mm"
name = "I8MM"
descr = "integer matrix multiplication"


def html():
    from markupsafe import Markup
    return Markup(f"""
<p>You are given an m × k matrix and a k × n matrix consisting of 8-bit integers. Your task is to calculate the m × n product of the two matrices.</p>

<h3>Interface</h3>
<p>You need to implement the following function:</p>
<div class="prewrap"><pre>
void gemm(int m, int n, int k, const int8_t* A, const int8_t* B, int32_t* C);
</pre></div>
<p>Here <code>A</code> and <code>B</code> are pointers to the input matrices, with <code>m</code> rows and <code>k</code> columns for <code>A</code>, and
<code>k</code> rows and <code>n</code> columns for <code>B</code>. 
For all <code>0 &lt;= y &lt; m</code> and <code>0 &lt;= x &lt; k</code>, the element at row <code>y</code> and column <code>x</code> of matrix <code>A</code> is stored in <code>A[x + y*k]</code>.
For all <code>0 &lt;= y &lt; k</code> and <code>0 &lt;= x &lt; n</code>, the element at row <code>y</code> and column <code>x</code> of matrix <code>B</code> is stored in <code>B[x + y*n]</code>.
</p>

<p>The function has to solve the following task: 
for all <code>i</code> and <code>j</code> with <code>0 &lt;= i &lt; m</code> and <code>0 &lt;= j &lt; n</code>, 
calculate the inner product between row <code>i</code> of A and column <code>j</code> of the B, and store the result in <code>C[j + i*n]</code>.</p>

<p>The arrays <code>data</code> and <code>result</code> are already allocated by whoever calls this function; you do not need to do any memory management related to these arrays. 
For the tasks that are to be solved on the CPU, <code>A</code>, <code>B</code>, and <code>C</code> point to CPU memory,
for tasks to be solved on the GPU they point to device memory.
You should not assume that <code>C</code> contains any valid values at the point of call. In particular, it is not guaranteed to be initialized with zeros.
</p>

<h3>Details</h3>
<p>The reduction dimension <code>k</code> is guaranteed to be less than 65536, so that all results can be represented as 32-bit signed integers.</p>

<p>
While floating-point and integer matrix multiplication appear very similar, at the mirco-architectural level,
there is one crucial difference: When multiplying two 32-bit floating-point numbers, the result is again a 32-bit floating-point
number, that can be added to a 32-bit floating-point number. In contrast, the product of two 8-bit integers is a 16-bit 
integer, and if you want to add multiple of these products, the accumulator needs to be a 32-bit integer.
</p>

<p>
There <em>cannot</em> be a SIMD instruction that takes two vector-registers of packed 8-bit integers and 
accumulates to a third register (like, e.g., <code>VFMADDPS</code>); the destination register is simply to small
to accumulate all 64 products, assuming 512-bit wide registers. Instead, the hardware implements inner-product
like operations: Take pairs (or groups of 4) of 8-bit integers in one operand, multiply each with the corresponding
8-bit integer in the second operand, sum the individual products and accumulate into the destination operand. This way,
the destination can contain fewer, but higher bit-width integers.
</p>

<h3 id=hint>SIMD Hints</h3>
<div class="spoiler">
<p>
In generic AVX-512, there is one instruction for doing an 8-bit inner product over pairs of numbers with 16-bit accumulation.
This is not particularly useful, because accumulation needs to happen in 32 bits to prevent overflows. 
However, a similar instruction exists for an inner product over two 16-bit numbers, with accumulation in 32 bit.
Expanding the 8-bit numbers to 16 bit and then using <code>_mm512_madd_epi16</code> can be a viable strategy.
</p>
<p>
For the VNNI task, note that the available instruction <code>_mm512_dpbusds_epi32</code> <em>only</em> 
allows multiplying one signed operand and one unsigned operand. In order to reap the speed benefits of this instruction,
you thus need implement pre- and postprocessing that maps signed integer matrix multiplication to signed time unsigned
matrix multiplication. The <code>__dp4a</code> intrinsic in CUDA directly supports signed times signed multiplication.
</p>
</div>

<h3 id=hint>Tensorcore Hints</h3>

<div class="spoiler">
<p>
A simple mental model for the basic operation of a tensorcore is that it extends the vector operations of regular SIMD
processing to instructions that operate on fixed-size matrix fragments. When using  8-bit integer operands, each <em>warp</em>
of the GPU can process the product of two <em>16 × 16</em> fragments in a single instruction. The C++ interface to these
instructions is documented in the <a href="https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#warp-matrix-functions">Cuda Programming Guide</a>.
</p>
<p>
As such, you can consider the input and output matrices as built up out of 16 × 16 tiles, and the algorithm can be implemented
the same way as a scalar matrix multiplication, except each element is now a matrix fragment. In particular, optimizations
like register reuse (now on the level of entire fragments), shared memory,  and the choice of the right data layout,
remain critical for good performance.
</p>
</div>


    """)


def explain_web(raw: dict):
    return generate_html(explain(json_to_output(raw), "web"))


def explain_terminal(r, color=False):
    printer = TerminalPrinter(color=color)
    if color:
        hl, minor, reset = '\033[31;1m', '\033[34;1m', '\033[0m'
        printer.set_format("tile correct", minor, reset)
        printer.set_format("tile verywrong", hl, reset)
    else:
        printer.set_format("tile correct", ' ', ' ')
        printer.set_format("tile verywrong", '[', ']')

    return generate_term(explain(r, "term"), printer)


def explain(r, mode: str):
    builder = DocumentBuilder(mode)

    input = r.input_data or {}
    output = r.output_data or {}
    oe = r.output_errors or {}

    m = input.get('m', None)
    n = input.get('n', None)
    k = input.get('k', None)
    tile = input.get('tile_size', None)
    a = input.get('input_a', None)
    b = input.get('input_b', None)
    result = output.get("result", None)
    locations = oe.get("locations", None)

    if m is not None and n is not None and k is not None:
        with builder.text() as txt:
            txt += f'In this test I called your function with the following parameters:\n'
        with builder.list(style="compact") as lst:
            lst.add_item(f'm = {m}')
            lst.add_item(f'n = {n}')
            lst.add_item(f'k = {k}')

        if tile is not None:
            m //= tile
            n //= tile
            k //= tile

        if a is not None and b is not None:
            if tile is not None:
                with builder.text() as txt:
                    txt += f'The input consisted of block matrices with blocks of size {tile} × {tile}.\n'
                    txt += f'Each number below indicates an entire {tile} × {tile} submatrix with constant coefficients.\n'
                    txt += f'Outputs have been divided by {tile}.\n\n'

            with builder.text() as txt:
                txt += f'This is what the input data looked like:\n'

            with builder.list(style="compact") as lst:
                lst.add_item('A')

            with builder.matrix(m, k) as mat:
                for y in range(m):
                    for x in range(k):
                        mat.entry(y, x, safeprint(safeget(a, y, x)))

            with builder.list(style="compact") as lst:
                lst.add_item('B')

            with builder.matrix(k, n) as mat:  # type: MatrixBuilder
                for y in range(k):
                    for x in range(n):
                        mat.entry(y, x, safeprint(safeget(b, y, x)))

        if result is not None:
            with builder.text() as txt:
                txt += f'This is the output that I got back:\n'
            with builder.matrix(m, n) as mat:
                for y in range(m):
                    for x in range(n):
                        v = safeget(result, y, x)
                        if locations:
                            e = safeget(locations, y, x)
                            if e == 0:
                                mat.entry(y, x, safeprint(v), style="correct")
                            else:
                                mat.entry(y,
                                          x,
                                          safeprint(v),
                                          style="verywrong")
            if locations is not None:
                with builder.text() as txt:
                    txt += f'Above I have highlighted the cells that contain wrong values\n'
        elif locations is not None:
            with builder.text() as txt:
                txt += f'This is the pattern of correct and incorrect results I got back:\n'
            good = StringNode('·', style="tile correct")
            bad = StringNode('×', style="tile verywrong")
            if mode == "web":
                with builder.matrix(m, n) as mat:
                    for y in range(m):
                        for x in range(n):
                            e = safeget(locations, y, x)
                            if e == 0:
                                mat.entry(y, x, "·", style="correct")
                            else:
                                mat.entry(y, x, "×", style="verywrong")
            else:
                with builder.text() as txt:
                    for y in range(m):
                        txt += ' '
                        for x in range(n):
                            e = safeget(locations, y, x)
                            if e == 0:
                                txt += good
                            else:
                                txt += bad
                        txt += '\n'
                    txt += '\n'
            with builder.text() as txt:
                txt += f'Above I have highlighted the cells as follows:\n'
            with builder.list(style="compact") as lst:
                lst.add_item(good + ' — correct result')
                lst.add_item(bad + ' — wrong result')
        else:
            with builder.text() as txt:
                txt += f'The probabilistic tests determined that there is an error in your result.'

    return builder.build()
