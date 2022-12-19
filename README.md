[![CodeQL](https://github.com/stefan-zobel/JAMU/actions/workflows/codeql.yml/badge.svg)](https://github.com/stefan-zobel/JAMU/actions/workflows/codeql.yml)
[![Maven Central](https://maven-badges.herokuapp.com/maven-central/net.sourceforge.streamsupport/jamu/badge.svg)](https://maven-badges.herokuapp.com/maven-central/net.sourceforge.streamsupport/jamu)
[![javadoc.io](https://javadoc.io/badge2/net.sourceforge.streamsupport/jamu/javadoc.svg)](https://javadoc.io/doc/net.sourceforge.streamsupport/jamu)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

# JAMU

JAMU - Java Matrix Utilities built on top of the [dedekind-MKL](https://github.com/stefan-zobel/dedekind-MKL) wrapper

*JAMU* has its focus on general dense matrices, there is no dedicated support for sparse matrices or special matrix structures like symmetric, triangular, (tri-)diagonal, banded or block matrices. 

Unlike many other Java matrix libraries, *JAMU* supports real ([MatrixD](https://github.com/stefan-zobel/JAMU/blob/master/src/main/java/net/jamu/matrix/MatrixD.java) and [MatrixF](https://github.com/stefan-zobel/JAMU/blob/master/src/main/java/net/jamu/matrix/MatrixF.java)) as well as complex matrices ([ComplexMatrixD](https://github.com/stefan-zobel/JAMU/blob/master/src/main/java/net/jamu/matrix/ComplexMatrixD.java) / [ComplexMatrixF](https://github.com/stefan-zobel/JAMU/blob/master/src/main/java/net/jamu/matrix/ComplexMatrixF.java)) for both single (`float` => `F` suffix) and double (`double` => `D` suffix) precision. The API is organized in 4 parallel (independent) inheritance hierarchies of which each provides the same methods. Together with the sole utility class [Matrices](https://github.com/stefan-zobel/JAMU/blob/master/src/main/java/net/jamu/matrix/Matrices.java) this offers a no frills API that is easy to use. *JAMU* doesn't provide distinguished *vectors*, whenever you want to work with vectors you should use a `n x 1` (column vector) matrix or `1 x n` (row vector) matrix instead.  

As to the supported matrix operations, apart from the usual suspects that each matrix library has to offer, the `LU`, `QR`, `EVD` and `SVD` decompositions are covered. Additionally, the Moore-Penrose `Pseudo-Inverse`, matrix exponentials (`expm`), `mldivide`, `mrdivide`, a couple of matrix norms, (de-)serialization of matrices and functions for the distance and approximate equality of matrices are also provided. Just give it a go.  


### Matrix size limitations

Matrices in *JAMU* are internally backed by 1-dimensional Java arrays in column-major storage layout which get passed to the *C* BLAS / LAPACK routines from Intel MKL (in a no-copy fashion). As such, the total size of a matrix is constrained by the maximum length of a Java array. In other words, if your matrix dimension `m x n` gets beyond `2^31 - 1` you can't use JAMU.   


### Where it shines

... is speed for not too small matrices. If you regularly work with `10 x 10` matrices use something else. If your matrix dimension is more like `1000 x 1000` or larger and you have lots of `Level 3` matrix operations (like matrix multiplication or matrix decompositions) there is nothing in the Java world which can beat the performance of the underlying MKL implementation. 


### Maven

```xml
<dependency>
    <groupId>net.sourceforge.streamsupport</groupId>
    <artifactId>jamu</artifactId>
    <version>1.3.0</version>
</dependency>
```


### Setup

*JAMU* depends on the [dedekind-MKL](https://github.com/stefan-zobel/dedekind-MKL) library which itself expects a functional Intel MKL installation. Turn to its [readme](https://github.com/stefan-zobel/dedekind-MKL/blob/master/README.md) for a description of where to get and how to setup the MKL libraries. 


### Credits

* the API is for the most part inspired by [JAMA](https://math.nist.gov/javanumerics/jama/) and [MTJ](https://github.com/fommil/matrix-toolkits-java) 
* the implementation owes somewhat to the [MTJ](https://github.com/fommil/matrix-toolkits-java) design
