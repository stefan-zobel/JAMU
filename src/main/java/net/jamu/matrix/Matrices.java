/*
 * Copyright 2019, 2020 Stefan Zobel
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package net.jamu.matrix;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Objects;
import java.util.Random;
import java.util.concurrent.ThreadLocalRandom;

import net.dedekind.blas.Blas;
import net.dedekind.lapack.Lapack;
import net.jamu.complex.ZArrayUtil;
import net.jamu.complex.Zd;
import net.jamu.complex.ZdImpl;
import net.jamu.complex.Zf;
import net.jamu.complex.ZfImpl;

/**
 * Static utility methods for matrices.
 */
public final class Matrices {

    /**
     * Create a new {@link MatrixD} of dimension {@code (row, cols)}.
     * 
     * @param rows
     *            number or rows
     * @param cols
     *            number of columns
     * @return a {@code MatrixD} of dimension {@code (row, cols)}
     */
    public static MatrixD createD(int rows, int cols) {
        return new SimpleMatrixD(rows, cols);
    }

    /**
     * Create a new {@link MatrixF} of dimension {@code (row, cols)}.
     * 
     * @param rows
     *            number or rows
     * @param cols
     *            number of columns
     * @return a {@code MatrixF} of dimension {@code (row, cols)}
     */
    public static MatrixF createF(int rows, int cols) {
        return new SimpleMatrixF(rows, cols);
    }

    /**
     * Create a new {@link ComplexMatrixD} of dimension {@code (row, cols)}.
     * 
     * @param rows
     *            number or rows
     * @param cols
     *            number of columns
     * @return a {@code ComplexMatrixD} of dimension {@code (row, cols)}
     */
    public static ComplexMatrixD createComplexD(int rows, int cols) {
        return new SimpleComplexMatrixD(rows, cols);
    }

    /**
     * Create a new {@link ComplexMatrixF} of dimension {@code (row, cols)}.
     * 
     * @param rows
     *            number or rows
     * @param cols
     *            number of columns
     * @return a {@code ComplexMatrixF} of dimension {@code (row, cols)}
     */
    public static ComplexMatrixF createComplexF(int rows, int cols) {
        return new SimpleComplexMatrixF(rows, cols);
    }

    /**
     * Create a {@link MatrixD} from a {@code double[][]} array. The elements of
     * {@code data} get copied, i.e. the array is not referenced.
     * <p>
     * The first index of {@code data} is interpreted as the row index. Note
     * that all rows must have the same length otherwise an
     * IllegalArgumentException is thrown.
     * 
     * @param data
     *            array whose shape and content determines the shape and content
     *            of the newly created matrix
     * @return a {@code MatrixD} of the same shape as {@code data} filled with
     *         the content of {@code data}.
     * @throws IllegalArgumentException
     *             if not all rows have the same length
     */
    public static MatrixD fromJaggedArrayD(double[][] data) {
        double[] copy = Checks.checkJaggedArrayD(data);
        int _rows = data.length;
        int _cols = data[0].length;
        for (int row = 0; row < _rows; ++row) {
            double[] row_i = data[row];
            if (row_i.length != _cols) {
                Checks.throwInconsistentRowLengths(_cols, row, row_i.length);
            }
            for (int col = 0; col < row_i.length; ++col) {
                copy[col * _rows + row] = row_i[col];
            }
        }
        return new SimpleMatrixD(_rows, _cols, copy);
    }

    /**
     * Create a {@link ComplexMatrixD} from a {@code double[][]} array of
     * complex numbers where each complex number is a pair of two doubles, first
     * the real part and then the imaginary part. The elements of
     * {@code complexdata} get copied, i.e. the array is not referenced.
     * <p>
     * The first index of {@code complexdata} is interpreted as the row index.
     * Note that all rows must have the same length (which equals twice the
     * number of columns in the matrix) otherwise an IllegalArgumentException is
     * thrown.
     * 
     * @param complexdata
     *            array whose shape and content determines the shape and content
     *            of the newly created matrix
     * @return a {@code ComplexMatrixD} of the same shape as {@code complexdata}
     *         filled with the content of {@code complexdata}.
     * @throws IllegalArgumentException
     *             if not all rows have the same length or if that length is not
     *             an even number
     */
    public static ComplexMatrixD fromJaggedComplexArrayD(double[][] complexdata) {
        double[] copy = Checks.checkJaggedComplexArrayD(complexdata);
        int _rows = complexdata.length;
        int _cols = complexdata[0].length;
        for (int row = 0; row < _rows; ++row) {
            double[] row_i = complexdata[row];
            if (row_i.length != _cols) {
                Checks.throwInconsistentRowLengths(_cols, row, row_i.length);
            }
            for (int col = 0; col < row_i.length; col += 2) {
                int i = 2 * ((col / 2) * _rows + row);
                copy[i] = row_i[col];
                copy[i + 1] = row_i[col + 1]; // "lgtm[java/index-out-of-bounds]"
                
            }
        }
        return new SimpleComplexMatrixD(_rows, _cols / 2, copy);
    }

    /**
     * Create a {@link MatrixF} from a {@code float[][]} array. The elements of
     * {@code data} get copied, i.e. the array is not referenced.
     * <p>
     * The first index of {@code data} is interpreted as the row index. Note
     * that all rows must have the same length otherwise an
     * IllegalArgumentException is thrown.
     * 
     * @param data
     *            array whose shape and content determines the shape and content
     *            of the newly created matrix
     * @return a {@code MatrixF} of the same shape as {@code data} filled with
     *         the content of {@code data}.
     * @throws IllegalArgumentException
     *             if not all rows have the same length
     */
    public static MatrixF fromJaggedArrayF(float[][] data) {
        float[] copy = Checks.checkJaggedArrayF(data);
        int _rows = data.length;
        int _cols = data[0].length;
        for (int row = 0; row < _rows; ++row) {
            float[] row_i = data[row];
            if (row_i.length != _cols) {
                Checks.throwInconsistentRowLengths(_cols, row, row_i.length);
            }
            for (int col = 0; col < row_i.length; ++col) {
                copy[col * _rows + row] = row_i[col];
            }
        }
        return new SimpleMatrixF(_rows, _cols, copy);
    }

    /**
     * Create a {@link ComplexMatrixF} from a {@code float[][]} array of complex
     * numbers where each complex number is a pair of two floats, first the real
     * part and then the imaginary part. The elements of {@code complexdata} get
     * copied, i.e. the array is not referenced.
     * <p>
     * The first index of {@code complexdata} is interpreted as the row index.
     * Note that all rows must have the same length (which equals twice the
     * number of columns in the matrix) otherwise an IllegalArgumentException is
     * thrown.
     * 
     * @param complexdata
     *            array whose shape and content determines the shape and content
     *            of the newly created matrix
     * @return a {@code ComplexMatrixF} of the same shape as {@code complexdata}
     *         filled with the content of {@code complexdata}.
     * @throws IllegalArgumentException
     *             if not all rows have the same length or if that length is not
     *             an even number
     */
    public static ComplexMatrixF fromJaggedComplexArrayF(float[][] complexdata) {
        float[] copy = Checks.checkJaggedComplexArrayF(complexdata);
        int _rows = complexdata.length;
        int _cols = complexdata[0].length;
        for (int row = 0; row < _rows; ++row) {
            float[] row_i = complexdata[row];
            if (row_i.length != _cols) {
                Checks.throwInconsistentRowLengths(_cols, row, row_i.length);
            }
            for (int col = 0; col < row_i.length; col += 2) {
                int i = 2 * ((col / 2) * _rows + row);
                copy[i] = row_i[col];
                copy[i + 1] = row_i[col + 1]; // "lgtm[java/index-out-of-bounds]"
                
            }
        }
        return new SimpleComplexMatrixF(_rows, _cols / 2, copy);
    }

    /**
     * Create an MatrixD identity matrix of dimension {@code (n, n)}.
     * 
     * @param n
     *            dimension of the quadratic identity matrix
     * @return MatrixD identity matrix of dimension {@code (n, n)}
     */
    public static MatrixD identityD(int n) {
        SimpleMatrixD m = new SimpleMatrixD(n, n);
        for (int i = 0; i < n; ++i) {
            m.set(i, i, 1.0);
        }
        return m;
    }

    /**
     * Create an MatrixF identity matrix of dimension {@code (n, n)}.
     * 
     * @param n
     *            dimension of the quadratic identity matrix
     * @return MatrixF identity matrix of dimension {@code (n, n)}
     */
    public static MatrixF identityF(int n) {
        SimpleMatrixF m = new SimpleMatrixF(n, n);
        for (int i = 0; i < n; ++i) {
            m.set(i, i, 1.0f);
        }
        return m;
    }

    /**
     * Create an ComplexMatrixD identity matrix of dimension {@code (n, n)}.
     * 
     * @param n
     *            dimension of the quadratic identity matrix
     * @return ComplexMatrixD identity matrix of dimension {@code (n, n)}
     */
    public static ComplexMatrixD identityComplexD(int n) {
        ComplexMatrixD m = new SimpleComplexMatrixD(n, n);
        for (int i = 0; i < n; ++i) {
            m.set(i, i, 1.0, 0.0);
        }
        return m;
    }

    /**
     * Create an ComplexMatrixF identity matrix of dimension {@code (n, n)}.
     * 
     * @param n
     *            dimension of the quadratic identity matrix
     * @return ComplexMatrixF identity matrix of dimension {@code (n, n)}
     */
    public static ComplexMatrixF identityComplexF(int n) {
        ComplexMatrixF m = new SimpleComplexMatrixF(n, n);
        for (int i = 0; i < n; ++i) {
            m.set(i, i, 1.0f, 0.0f);
        }
        return m;
    }

    /**
     * Create a quadratic diagonal matrix whose main diagonal contains the
     * entries provided in the {@code diagonal} array.
     * 
     * @param diagonal
     *            the entries to be copied to the main diagonal. Must not be
     *            {@code null} and must have length {@code > 0}
     * @return a {@code diagonal.length x diagonal.length} diagonal matrix with
     *         its main diagonal entries equal to the elements from the provided
     *         array
     */
    public static MatrixD diagD(double[] diagonal) {
        if (Objects.requireNonNull(diagonal).length == 0) {
            throw new IllegalArgumentException("diagonal array length must be > 0");
        }
        int n = diagonal.length;
        SimpleMatrixD m = new SimpleMatrixD(n, n);
        for (int i = 0; i < n; ++i) {
            m.set(i, i, diagonal[i]);
        }
        return m;
    }

    /**
     * Create a quadratic diagonal matrix of dimension {@code (n, n)} where all
     * the main diagonal entries contain the value {@code diagVal}.
     * 
     * @param n
     *            the dimension {@code (n, n)} of the quadratic matrix
     * @param diagVal
     *            the value to which all main diagonal entries are initialized
     * @return a {@code (n, n)} diagonal matrix with its main diagonal entries
     *         equal to {@code diagVal}
     */
    public static MatrixD diagD(int n, double diagVal) {
        SimpleMatrixD m = new SimpleMatrixD(n, n);
        for (int i = 0; i < n; ++i) {
            m.set(i, i, diagVal);
        }
        return m;
    }

    /**
     * Create a quadratic diagonal matrix whose main diagonal contains the
     * entries provided in the {@code diagonal} array.
     * 
     * @param diagonal
     *            the entries to be copied to the main diagonal. Must not be
     *            {@code null} and must have length {@code > 0}
     * @return a {@code diagonal.length x diagonal.length} diagonal matrix with
     *         its main diagonal entries equal to the elements from the provided
     *         array
     */
    public static MatrixF diagF(float[] diagonal) {
        if (Objects.requireNonNull(diagonal).length == 0) {
            throw new IllegalArgumentException("diagonal array length must be > 0");
        }
        int n = diagonal.length;
        SimpleMatrixF m = new SimpleMatrixF(n, n);
        for (int i = 0; i < n; ++i) {
            m.set(i, i, diagonal[i]);
        }
        return m;
    }

    /**
     * Create a quadratic diagonal matrix of dimension {@code (n, n)} where all
     * the main diagonal entries contain the value {@code diagVal}.
     * 
     * @param n
     *            the dimension {@code (n, n)} of the quadratic matrix
     * @param diagVal
     *            the value to which all main diagonal entries are initialized
     * @return a {@code (n, n)} diagonal matrix with its main diagonal entries
     *         equal to {@code diagVal}
     */
    public static MatrixF diagF(int n, float diagVal) {
        SimpleMatrixF m = new SimpleMatrixF(n, n);
        for (int i = 0; i < n; ++i) {
            m.set(i, i, diagVal);
        }
        return m;
    }

    /**
     * Create a quadratic diagonal matrix whose main diagonal contains the
     * entries provided in the {@code diagonal} array.
     * 
     * @param diagonal
     *            the entries to be copied to the main diagonal. Must not be
     *            {@code null} and must have length {@code > 0}
     * @return a {@code diagonal.length x diagonal.length} diagonal matrix with
     *         its main diagonal entries equal to the elements from the provided
     *         array
     */
    public static ComplexMatrixD diagComplexD(Zd[] diagonal) {
        if (Objects.requireNonNull(diagonal).length == 0) {
            throw new IllegalArgumentException("diagonal array length must be > 0");
        }
        int n = diagonal.length;
        SimpleComplexMatrixD m = new SimpleComplexMatrixD(n, n);
        for (int i = 0; i < n; ++i) {
            Zd z = diagonal[i];
            if (z != null) {
                m.set(i, i, z.re(), z.im());
            }
        }
        return m;
    }

    /**
     * Create a quadratic diagonal matrix of dimension {@code (n, n)} where all
     * the main diagonal entries contain the complex value with real and
     * imaginary parts {@code (diagReVal, diagImVal)}.
     * 
     * @param n
     *            the dimension {@code (n, n)} of the quadratic matrix
     * @param diagReVal
     *            the real part of the complex number to which all main diagonal
     *            entries are initialized
     * @param diagImVal
     *            the imaginary part of the complex number to which all main
     *            diagonal entries are initialized
     * @return a diagonal matrix of dimension {@code (n, n)} with its main
     *         diagonal entries equal to the complex number with real and
     *         imaginary parts represented by the pair
     *         {@code (diagReVal, diagImVal)}
     */
    public static ComplexMatrixD diagComplexD(int n, double diagReVal, double diagImVal) {
        SimpleComplexMatrixD m = new SimpleComplexMatrixD(n, n);
        for (int i = 0; i < n; ++i) {
            m.set(i, i, diagReVal, diagImVal);
        }
        return m;
    }

    /**
     * Create a quadratic diagonal matrix whose main diagonal contains the
     * entries provided in the {@code diagonal} array.
     * 
     * @param diagonal
     *            the entries to be copied to the main diagonal. Must not be
     *            {@code null} and must have length {@code > 0}
     * @return a {@code diagonal.length x diagonal.length} diagonal matrix with
     *         its main diagonal entries equal to the elements from the provided
     *         array
     */
    public static ComplexMatrixF diagComplexF(Zf[] diagonal) {
        if (Objects.requireNonNull(diagonal).length == 0) {
            throw new IllegalArgumentException("diagonal array length must be > 0");
        }
        int n = diagonal.length;
        SimpleComplexMatrixF m = new SimpleComplexMatrixF(n, n);
        for (int i = 0; i < n; ++i) {
            Zf z = diagonal[i];
            if (z != null) {
                m.set(i, i, z.re(), z.im());
            }
        }
        return m;
    }

    /**
     * Create a quadratic diagonal matrix of dimension {@code (n, n)} where all
     * the main diagonal entries contain the complex value with real and
     * imaginary parts {@code (diagReVal, diagImVal)}.
     * 
     * @param n
     *            the dimension {@code (n, n)} of the quadratic matrix
     * @param diagReVal
     *            the real part of the complex number to which all main diagonal
     *            entries are initialized
     * @param diagImVal
     *            the imaginary part of the complex number to which all main
     *            diagonal entries are initialized
     * @return a diagonal matrix of dimension {@code (n, n)} with its main
     *         diagonal entries equal to the complex number with real and
     *         imaginary parts represented by the pair
     *         {@code (diagReVal, diagImVal)}
     */
    public static ComplexMatrixF diagComplexF(int n, float diagReVal, float diagImVal) {
        SimpleComplexMatrixF m = new SimpleComplexMatrixF(n, n);
        for (int i = 0; i < n; ++i) {
            m.set(i, i, diagReVal, diagImVal);
        }
        return m;
    }

    /**
     * Create a MatrixD of dimension {@code (rows, cols)} filled with uniformly
     * distributed random numbers drawn from the range {@code [0.0, 1.0)}.
     * 
     * @param rows
     *            number or rows
     * @param cols
     *            number or columns
     * @return {@code (rows, cols)} MatrixD filled with {@code ~U[0, 1]}
     *         distributed random numbers
     */
    public static MatrixD randomUniformD(int rows, int cols) {
        return randomUniformD(rows, cols, null);
    }

    /**
     * Create a MatrixD of dimension {@code (rows, cols)} filled with uniformly
     * distributed random numbers drawn from the range {@code [0.0, 1.0)}.
     * 
     * @param rows
     *            number or rows
     * @param cols
     *            number or columns
     * @param seed
     *            the initial seed to use for the PRNG
     * @return {@code (rows, cols)} MatrixD filled with {@code ~U[0, 1]}
     *         distributed random numbers
     */
    public static MatrixD randomUniformD(int rows, int cols, long seed) {
        return randomUniformD(rows, cols, new Random(seed));
    }

    private static MatrixD randomUniformD(int rows, int cols, Random rng) {
        SimpleMatrixD m = new SimpleMatrixD(rows, cols);
        Random rnd = (rng == null) ? ThreadLocalRandom.current() : rng;
        double[] _a = m.getArrayUnsafe();
        for (int i = 0; i < _a.length; ++i) {
            _a[i] = rnd.nextDouble();
        }
        return m;
    }

    /**
     * Create a MatrixF of dimension {@code (rows, cols)} filled with uniformly
     * distributed random numbers drawn from the range {@code [0.0f, 1.0f)}.
     * 
     * @param rows
     *            number or rows
     * @param cols
     *            number or columns
     * @return {@code (rows, cols)} MatrixF filled with {@code ~U[0, 1]}
     *         distributed random numbers
     */
    public static MatrixF randomUniformF(int rows, int cols) {
        return randomUniformF(rows, cols, null);
    }

    /**
     * Create a MatrixF of dimension {@code (rows, cols)} filled with uniformly
     * distributed random numbers drawn from the range {@code [0.0f, 1.0f)}.
     * 
     * @param rows
     *            number or rows
     * @param cols
     *            number or columns
     * @param seed
     *            the initial seed to use for the PRNG
     * @return {@code (rows, cols)} MatrixF filled with {@code ~U[0, 1]}
     *         distributed random numbers
     */
    public static MatrixF randomUniformF(int rows, int cols, long seed) {
        return randomUniformF(rows, cols, new Random(seed));
    }

    private static MatrixF randomUniformF(int rows, int cols, Random rng) {
        SimpleMatrixF m = new SimpleMatrixF(rows, cols);
        Random rnd = (rng == null) ? ThreadLocalRandom.current() : rng;
        float[] _a = m.getArrayUnsafe();
        for (int i = 0; i < _a.length; ++i) {
            _a[i] = rnd.nextFloat();
        }
        return m;
    }

    /**
     * Create a ComplexMatrixD of dimension {@code (rows, cols)} filled with
     * uniformly distributed random complex numbers drawn from the range
     * {@code [0.0, 1.0)} for both the real part and the imaginary part.
     * 
     * @param rows
     *            number or rows
     * @param cols
     *            number or columns
     * @return {@code (rows, cols)} ComplexMatrixD filled with {@code ~U[0, 1]}
     *         distributed random complex numbers where {@code ~U[0, 1]}
     *         pertains to the real and imaginary part individually
     */
    public static ComplexMatrixD randomUniformComplexD(int rows, int cols) {
        return randomUniformComplexD(rows, cols, null);
    }

    /**
     * Create a ComplexMatrixD of dimension {@code (rows, cols)} filled with
     * uniformly distributed random complex numbers drawn from the range
     * {@code [0.0, 1.0)} for both the real part and the imaginary part.
     * 
     * @param rows
     *            number or rows
     * @param cols
     *            number or columns
     * @param seed
     *            the initial seed to use for the PRNG
     * @return {@code (rows, cols)} ComplexMatrixD filled with {@code ~U[0, 1]}
     *         distributed random complex numbers where {@code ~U[0, 1]}
     *         pertains to the real and imaginary part individually
     */
    public static ComplexMatrixD randomUniformComplexD(int rows, int cols, long seed) {
        return randomUniformComplexD(rows, cols, new Random(seed));
    }

    private static ComplexMatrixD randomUniformComplexD(int rows, int cols, Random rng) {
        SimpleComplexMatrixD m = new SimpleComplexMatrixD(rows, cols);
        Random rnd = (rng == null) ? ThreadLocalRandom.current() : rng;
        double[] _a = m.getArrayUnsafe();
        for (int i = 0; i < _a.length; ++i) {
            _a[i] = rnd.nextDouble();
        }
        return m;
    }

    /**
     * Create a ComplexMatrixF of dimension {@code (rows, cols)} filled with
     * uniformly distributed random complex numbers drawn from the range
     * {@code [0.0f, 1.0f)} for both the real part and the imaginary part.
     * 
     * @param rows
     *            number or rows
     * @param cols
     *            number or columns
     * @return {@code (rows, cols)} ComplexMatrixF filled with {@code ~U[0, 1]}
     *         distributed random complex numbers where {@code ~U[0, 1]}
     *         pertains to the real and imaginary part individually
     */
    public static ComplexMatrixF randomUniformComplexF(int rows, int cols) {
        return randomUniformComplexF(rows, cols, null);
    }

    /**
     * Create a ComplexMatrixF of dimension {@code (rows, cols)} filled with
     * uniformly distributed random complex numbers drawn from the range
     * {@code [0.0f, 1.0f)} for both the real part and the imaginary part.
     * 
     * @param rows
     *            number or rows
     * @param cols
     *            number or columns
     * @param seed
     *            the initial seed to use for the PRNG
     * @return {@code (rows, cols)} ComplexMatrixF filled with {@code ~U[0, 1]}
     *         distributed random complex numbers where {@code ~U[0, 1]}
     *         pertains to the real and imaginary part individually
     */
    public static ComplexMatrixF randomUniformComplexF(int rows, int cols, long seed) {
        return randomUniformComplexF(rows, cols, new Random(seed));
    }

    private static ComplexMatrixF randomUniformComplexF(int rows, int cols, Random rng) {
        SimpleComplexMatrixF m = new SimpleComplexMatrixF(rows, cols);
        Random rnd = (rng == null) ? ThreadLocalRandom.current() : rng;
        float[] _a = m.getArrayUnsafe();
        for (int i = 0; i < _a.length; ++i) {
            _a[i] = rnd.nextFloat();
        }
        return m;
    }

    /**
     * Create a MatrixD of dimension {@code (rows, cols)} filled with normally
     * distributed (i.e., standard gausssian) random numbers with expectation
     * {@code 0.0} and variance {@code 1.0}.
     * 
     * @param rows
     *            number or rows
     * @param cols
     *            number or columns
     * @return {@code (rows, cols)} MatrixD filled with {@code ~N[0, 1]}
     *         distributed random numbers (standard normal distribution)
     */
    public static MatrixD randomNormalD(int rows, int cols) {
        return randomNormalD(rows, cols, null);
    }

    /**
     * Create a MatrixD of dimension {@code (rows, cols)} filled with normally
     * distributed (i.e., standard gausssian) random numbers with expectation
     * {@code 0.0} and variance {@code 1.0}.
     * 
     * @param rows
     *            number or rows
     * @param cols
     *            number or columns
     * @param seed
     *            the initial seed to use for the PRNG
     * @return {@code (rows, cols)} MatrixD filled with {@code ~N[0, 1]}
     *         distributed random numbers (standard normal distribution)
     */
    public static MatrixD randomNormalD(int rows, int cols, long seed) {
        return randomNormalD(rows, cols, new Random(seed));
    }

    private static MatrixD randomNormalD(int rows, int cols, Random rng) {
        SimpleMatrixD m = new SimpleMatrixD(rows, cols);
        Random rnd = (rng == null) ? ThreadLocalRandom.current() : rng;
        double[] _a = m.getArrayUnsafe();
        for (int i = 0; i < _a.length; ++i) {
            _a[i] = rnd.nextGaussian();
        }
        return m;
    }

    /**
     * Create a MatrixF of dimension {@code (rows, cols)} filled with normally
     * distributed (i.e., standard gausssian) random numbers with expectation
     * {@code 0.0f} and variance {@code 1.0f}.
     * 
     * @param rows
     *            number or rows
     * @param cols
     *            number or columns
     * @return {@code (rows, cols)} MatrixF filled with {@code ~N[0, 1]}
     *         distributed random numbers (standard normal distribution)
     */
    public static MatrixF randomNormalF(int rows, int cols) {
        return randomNormalF(rows, cols, null);
    }

    /**
     * Create a MatrixF of dimension {@code (rows, cols)} filled with normally
     * distributed (i.e., standard gausssian) random numbers with expectation
     * {@code 0.0f} and variance {@code 1.0f}.
     * 
     * @param rows
     *            number or rows
     * @param cols
     *            number or columns
     * @param seed
     *            the initial seed to use for the PRNG
     * @return {@code (rows, cols)} MatrixF filled with {@code ~N[0, 1]}
     *         distributed random numbers (standard normal distribution)
     */
    public static MatrixF randomNormalF(int rows, int cols, long seed) {
        return randomNormalF(rows, cols, new Random(seed));
    }

    private static MatrixF randomNormalF(int rows, int cols, Random rng) {
        SimpleMatrixF m = new SimpleMatrixF(rows, cols);
        Random rnd = (rng == null) ? ThreadLocalRandom.current() : rng;
        float[] _a = m.getArrayUnsafe();
        for (int i = 0; i < _a.length; ++i) {
            _a[i] = (float) rnd.nextGaussian();
        }
        return m;
    }

    /**
     * Create a ComplexMatrixD of dimension {@code (rows, cols)} filled with
     * normally distributed (i.e., standard gausssian) random complex numbers
     * with expectation {@code 0.0} and variance {@code 1.0} for both the real
     * part and the imaginary part.
     * 
     * @param rows
     *            number or rows
     * @param cols
     *            number or columns
     * @return {@code (rows, cols)} ComplexMatrixD filled with {@code ~N[0, 1]}
     *         distributed random complex numbers (standard normal distribution)
     *         where {@code ~N[0, 1]} pertains to the real and imaginary part
     *         individually
     */
    public static ComplexMatrixD randomNormalComplexD(int rows, int cols) {
        return randomNormalComplexD(rows, cols, null);
    }

    /**
     * Create a ComplexMatrixD of dimension {@code (rows, cols)} filled with
     * normally distributed (i.e., standard gausssian) random complex numbers
     * with expectation {@code 0.0} and variance {@code 1.0} for both the real
     * part and the imaginary part.
     * 
     * @param rows
     *            number or rows
     * @param cols
     *            number or columns
     * @param seed
     *            the initial seed to use for the PRNG
     * @return {@code (rows, cols)} ComplexMatrixD filled with {@code ~N[0, 1]}
     *         distributed random complex numbers (standard normal distribution)
     *         where {@code ~N[0, 1]} pertains to the real and imaginary part
     *         individually
     */
    public static ComplexMatrixD randomNormalComplexD(int rows, int cols, long seed) {
        return randomNormalComplexD(rows, cols, new Random(seed));
    }

    private static ComplexMatrixD randomNormalComplexD(int rows, int cols, Random rng) {
        SimpleComplexMatrixD m = new SimpleComplexMatrixD(rows, cols);
        Random rnd = (rng == null) ? ThreadLocalRandom.current() : rng;
        double[] _a = m.getArrayUnsafe();
        for (int i = 0; i < _a.length; ++i) {
            _a[i] = rnd.nextGaussian();
        }
        return m;
    }

    /**
     * Create a ComplexMatrixF of dimension {@code (rows, cols)} filled with
     * normally distributed (i.e., standard gausssian) random complex numbers
     * with expectation {@code 0.0f} and variance {@code 1.0f} for both the real
     * part and the imaginary part.
     * 
     * @param rows
     *            number or rows
     * @param cols
     *            number or columns
     * @return {@code (rows, cols)} ComplexMatrixF filled with {@code ~N[0, 1]}
     *         distributed random complex numbers (standard normal distribution)
     *         where {@code ~N[0, 1]} pertains to the real and imaginary part
     *         individually
     */
    public static ComplexMatrixF randomNormalComplexF(int rows, int cols) {
        return randomNormalComplexF(rows, cols, null);
    }

    /**
     * Create a ComplexMatrixF of dimension {@code (rows, cols)} filled with
     * normally distributed (i.e., standard gausssian) random complex numbers
     * with expectation {@code 0.0f} and variance {@code 1.0f} for both the real
     * part and the imaginary part.
     * 
     * @param rows
     *            number or rows
     * @param cols
     *            number or columns
     * @param seed
     *            the initial seed to use for the PRNG
     * @return {@code (rows, cols)} ComplexMatrixF filled with {@code ~N[0, 1]}
     *         distributed random complex numbers (standard normal distribution)
     *         where {@code ~N[0, 1]} pertains to the real and imaginary part
     *         individually
     */
    public static ComplexMatrixF randomNormalComplexF(int rows, int cols, long seed) {
        return randomNormalComplexF(rows, cols, new Random(seed));
    }

    private static ComplexMatrixF randomNormalComplexF(int rows, int cols, Random rng) {
        SimpleComplexMatrixF m = new SimpleComplexMatrixF(rows, cols);
        Random rnd = (rng == null) ? ThreadLocalRandom.current() : rng;
        float[] _a = m.getArrayUnsafe();
        for (int i = 0; i < _a.length; ++i) {
            _a[i] = (float) rnd.nextGaussian();
        }
        return m;
    }

    /**
     * Create a MatrixD of dimension {@code (rows, cols)} whose elements are all
     * {@code 1.0}.
     * 
     * @param rows
     *            number or rows
     * @param cols
     *            number or columns
     * @return MatrixD of dimension {@code (rows, cols)} filled with ones
     */
    public static MatrixD onesD(int rows, int cols) {
        SimpleMatrixD m = new SimpleMatrixD(rows, cols);
        Arrays.fill(m.getArrayUnsafe(), 1.0);
        return m;
    }

    /**
     * Create a MatrixF of dimension {@code (rows, cols)} whose elements are all
     * {@code 1.0f}.
     * 
     * @param rows
     *            number or rows
     * @param cols
     *            number or columns
     * @return MatrixD of dimension {@code (rows, cols)} filled with ones
     */
    public static MatrixF onesF(int rows, int cols) {
        SimpleMatrixF m = new SimpleMatrixF(rows, cols);
        Arrays.fill(m.getArrayUnsafe(), 1.0f);
        return m;
    }

    /**
     * Create a MatrixD of dimension {@code (rows, cols)} filled with the
     * natural numbers starting with 1 in row-major order. This is mainly useful
     * for tests.
     * 
     * @param rows
     *            number or rows
     * @param cols
     *            number or columns
     * @return matrix filled with the natural numbers starting with 1 in
     *         row-major order
     */
    public static MatrixD naturalNumbersD(int rows, int cols) {
        SimpleMatrixD m = new SimpleMatrixD(rows, cols);
        int nat = 1;
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                m.set(i, j, nat++);
            }
        }
        return m;
    }

    /**
     * Create a MatrixF of dimension {@code (rows, cols)} filled with the
     * natural numbers starting with 1 in row-major order. This is mainly useful
     * for tests.
     * 
     * @param rows
     *            number or rows
     * @param cols
     *            number or columns
     * @return matrix filled with the natural numbers starting with 1 in
     *         row-major order
     */
    public static MatrixF naturalNumbersF(int rows, int cols) {
        SimpleMatrixF m = new SimpleMatrixF(rows, cols);
        int nat = 1;
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                m.set(i, j, nat++);
            }
        }
        return m;
    }

    /**
     * Create a new zero matrix of same dimension as {@code md}.
     * 
     * @param md
     *            {@code MatrixD} template for the dimensions to use for the new
     *            matrix
     * @return new zero matrix of same dimension as {@code md}
     */
    public static MatrixD sameDimD(MatrixD md) {
        return new SimpleMatrixD(md.numRows(), md.numColumns());
    }

    /**
     * Create a new zero matrix of same dimension as {@code mf}.
     * 
     * @param mf
     *            {@code MatrixF} template for the dimensions to use for the new
     *            matrix
     * @return new zero matrix of same dimension as {@code mf}
     */
    public static MatrixF sameDimF(MatrixF mf) {
        return new SimpleMatrixF(mf.numRows(), mf.numColumns());
    }

    /**
     * Create a new zero matrix of same dimension as {@code cmd}.
     * 
     * @param cmd
     *            {@code ComplexMatrixD} template for the dimensions to use for
     *            the new matrix
     * @return new zero matrix of same dimension as {@code cmd}
     */
    public static ComplexMatrixD sameDimComplexD(ComplexMatrixD cmd) {
        return new SimpleComplexMatrixD(cmd.numRows(), cmd.numColumns());
    }

    /**
     * Create a new zero matrix of same dimension as {@code cmf}.
     * 
     * @param cmf
     *            {@code ComplexMatrixF} template for the dimensions to use for
     *            the new matrix
     * @return new zero matrix of same dimension as {@code cmf}
     */
    public static ComplexMatrixF sameDimComplexF(ComplexMatrixF cmf) {
        return new SimpleComplexMatrixF(cmf.numRows(), cmf.numColumns());
    }

    /**
     * Writes the matrix {@code mf} into the provided Path {@code file} and
     * returns the number of bytes written.
     * 
     * @param mf
     *            the float matrix that needs to be serialized
     * @param file
     *            the file to write the {@code mf} matrix into
     * @return the number of bytes written into the file
     * @throws IOException
     *             if anything goes wrong
     */
    public static long serializeF(MatrixF mf, Path file) throws IOException {
        //@formatter:off
        try (OutputStream os = Files.newOutputStream(file);
             BufferedOutputStream bos = new BufferedOutputStream(os, BUF_SIZE)
        )
        {
            long sz = serializeF(mf, bos);
            bos.flush();
            return sz;
        }
        //@formatter:on
    }

    /**
     * Writes the complex matrix {@code cmf} into the provided Path {@code file}
     * and returns the number of bytes written.
     * 
     * @param cmf
     *            the complex float matrix that needs to be serialized
     * @param file
     *            the file to write the complex {@code cmf} matrix into
     * @return the number of bytes written into the file
     * @throws IOException
     *             if anything goes wrong
     */
    public static long serializeComplexF(ComplexMatrixF cmf, Path file) throws IOException {
        //@formatter:off
        try (OutputStream os = Files.newOutputStream(file);
             BufferedOutputStream bos = new BufferedOutputStream(os, BUF_SIZE)
        )
        {
            long sz = serializeComplexF(cmf, bos);
            bos.flush();
            return sz;
        }
        //@formatter:on
    }

    /**
     * Writes the matrix {@code mf} into the provided output stream {@code os}
     * and returns the number of bytes written. Everything related to (pre-)
     * positioning, flushing and closing of the output stream must be done by
     * the caller.
     * 
     * @param mf
     *            the float matrix that needs to be serialized
     * @param os
     *            the output stream to write the {@code mf} matrix into
     * @return the number of bytes written into the output stream
     * @throws IOException
     *             if anything goes wrong
     */
    public static long serializeF(MatrixF mf, OutputStream os) throws IOException {
        byte[] buf = new byte[4];
        long sz = IO.writeMatrixHeaderB(mf.numRows(), mf.numColumns(), Float.SIZE, buf, os);
        float[] data = mf.getArrayUnsafe();
        for (int i = 0; i < data.length; ++i) {
            sz += IO.putFloatB(data[i], buf, os);
        }
        return sz;
    }

    /**
     * Writes the complex matrix {@code cmf} into the provided output stream
     * {@code os} and returns the number of bytes written. Everything related to
     * (pre-) positioning, flushing and closing of the output stream must be
     * done by the caller.
     * 
     * @param cmf
     *            the complex float matrix that needs to be serialized
     * @param os
     *            the output stream to write the {@code cmf} matrix into
     * @return the number of bytes written into the output stream
     * @throws IOException
     *             if anything goes wrong
     */
    public static long serializeComplexF(ComplexMatrixF cmf, OutputStream os) throws IOException {
        byte[] buf = new byte[4];
        long sz = IO.writeMatrixHeaderB(cmf.numRows(), cmf.numColumns(), -Float.SIZE, buf, os);
        float[] data = cmf.getArrayUnsafe();
        for (int i = 0; i < data.length; ++i) {
            sz += IO.putFloatB(data[i], buf, os);
        }
        return sz;
    }

    /**
     * Writes the matrix {@code md} into the provided Path {@code file} and
     * returns the number of bytes written.
     * 
     * @param md
     *            the double matrix that needs to be serialized
     * @param file
     *            the file to write the {@code md} matrix into
     * @return the number of bytes written into the file
     * @throws IOException
     *             if anything goes wrong
     */
    public static long serializeD(MatrixD md, Path file) throws IOException {
        //@formatter:off
        try (OutputStream os = Files.newOutputStream(file);
             BufferedOutputStream bos = new BufferedOutputStream(os, BUF_SIZE)
        )
        {
            long sz = serializeD(md, bos);
            bos.flush();
            return sz;
        }
        //@formatter:on
    }

    /**
     * Writes the complex matrix {@code cmd} into the provided Path {@code file}
     * and returns the number of bytes written.
     * 
     * @param cmd
     *            the complex double matrix that needs to be serialized
     * @param file
     *            the file to write the complex {@code cmd} matrix into
     * @return the number of bytes written into the file
     * @throws IOException
     *             if anything goes wrong
     */
    public static long serializeComplexD(ComplexMatrixD cmd, Path file) throws IOException {
        //@formatter:off
        try (OutputStream os = Files.newOutputStream(file);
             BufferedOutputStream bos = new BufferedOutputStream(os, BUF_SIZE)
        )
        {
            long sz = serializeComplexD(cmd, bos);
            bos.flush();
            return sz;
        }
        //@formatter:on
    }

    /**
     * Writes the matrix {@code md} into the provided output stream {@code os}
     * and returns the number of bytes written. Everything related to (pre-)
     * positioning, flushing and closing of the output stream must be done by
     * the caller.
     * 
     * @param md
     *            the double matrix that needs to be serialized
     * @param os
     *            the output stream to write the {@code md} matrix into
     * @return the number of bytes written into the output stream
     * @throws IOException
     *             if anything goes wrong
     */
    public static long serializeD(MatrixD md, OutputStream os) throws IOException {
        byte[] buf = new byte[8];
        long sz = IO.writeMatrixHeaderB(md.numRows(), md.numColumns(), Double.SIZE, buf, os);
        double[] data = md.getArrayUnsafe();
        for (int i = 0; i < data.length; ++i) {
            sz += IO.putDoubleB(data[i], buf, os);
        }
        return sz;
    }

    /**
     * Writes the complex matrix {@code cmd} into the provided output stream
     * {@code os} and returns the number of bytes written. Everything related to
     * (pre-) positioning, flushing and closing of the output stream must be
     * done by the caller.
     * 
     * @param cmd
     *            the complex double matrix that needs to be serialized
     * @param os
     *            the output stream to write the {@code cmd} matrix into
     * @return the number of bytes written into the output stream
     * @throws IOException
     *             if anything goes wrong
     */
    public static long serializeComplexD(ComplexMatrixD cmd, OutputStream os) throws IOException {
        byte[] buf = new byte[8];
        long sz = IO.writeMatrixHeaderB(cmd.numRows(), cmd.numColumns(), -Double.SIZE, buf, os);
        double[] data = cmd.getArrayUnsafe();
        for (int i = 0; i < data.length; ++i) {
            sz += IO.putDoubleB(data[i], buf, os);
        }
        return sz;
    }
    /**
     * Attempts to read a float matrix from the provided Path {@code file}.
     * 
     * @param file
     *            the file to read the float matrix from
     * @return the deserialized float matrix
     * @throws IOException
     *             if anything goes wrong (e.g., there is no float matrix stored
     *             in that file)
     */
    public static MatrixF deserializeF(Path file) throws IOException {
        //@formatter:off
        try (InputStream is = Files.newInputStream(file);
             BufferedInputStream bis = new BufferedInputStream(is, BUF_SIZE)
        )
        {
            return deserializeF(bis);
        }
        //@formatter:on
    }

    /**
     * Attempts to read a complex float matrix from the provided Path
     * {@code file}.
     * 
     * @param file
     *            the file to read the complex float matrix from
     * @return the deserialized complex float matrix
     * @throws IOException
     *             if anything goes wrong (e.g., there is no complex float
     *             matrix stored in that file)
     */
    public static ComplexMatrixF deserializeComplexF(Path file) throws IOException {
        //@formatter:off
        try (InputStream is = Files.newInputStream(file);
             BufferedInputStream bis = new BufferedInputStream(is, BUF_SIZE)
        )
        {
            return deserializeComplexF(bis);
        }
        //@formatter:on
    }

    /**
     * Attempts to read a float matrix from the current position of the provided
     * input stream {@code is}. The correct positioning of the input stream and
     * closing the input stream must be done by the caller.
     * 
     * @param is
     *            the input stream to read a float matrix from
     * @return the float matrix deserialized from the input stream
     * @throws IOException
     *             if anything goes wrong (e.g., the position is wrong or there
     *             is no float matrix stored at the current position)
     */
    public static MatrixF deserializeF(InputStream is) throws IOException {
        byte[] buf = new byte[4];
        checkBigendian(IO.isBigendian(buf, is));
        if (IO.isDoubleType(buf, is)) {
            throw new IOException("Unexpected double type. Use double Deserializer instead.");
        }
        if (IO.isComplexType(buf)) {
            throw new IOException("Unexpected ComplexMatrixF. Use deserializeComplexF() instead.");
        }
        int rows = IO.readRows(true, buf, is);
        int cols = IO.readCols(true, buf, is);
        MatrixF mf = createF(rows, cols);
        float[] data = mf.getArrayUnsafe();
        for (int i = 0; i < data.length; ++i) {
            data[i] = IO.getFloatB(buf, is);
        }
        return mf;
    }

    /**
     * Attempts to read a complex float matrix from the current position of the
     * provided input stream {@code is}. The correct positioning of the input
     * stream and closing the input stream must be done by the caller.
     * 
     * @param is
     *            the input stream to read a complex float matrix from
     * @return the complex float matrix deserialized from the input stream
     * @throws IOException
     *             if anything goes wrong (e.g., the position is wrong or there
     *             is no complex float matrix stored at the current position)
     */
    public static ComplexMatrixF deserializeComplexF(InputStream is) throws IOException {
        byte[] buf = new byte[4];
        checkBigendian(IO.isBigendian(buf, is));
        if (IO.isDoubleType(buf, is)) {
            throw new IOException("Unexpected double type. Use double Deserializer instead.");
        }
        if (!IO.isComplexType(buf)) {
            throw new IOException("Unexpected MatrixF. Use deserializeF() instead.");
        }
        int rows = IO.readRows(true, buf, is);
        int cols = IO.readCols(true, buf, is);
        ComplexMatrixF cmf = createComplexF(rows, cols);
        float[] data = cmf.getArrayUnsafe();
        for (int i = 0; i < data.length; ++i) {
            data[i] = IO.getFloatB(buf, is);
        }
        return cmf;
    }

    /**
     * Attempts to read a double matrix from the provided Path {@code file}.
     * 
     * @param file
     *            the file to read the double matrix from
     * @return the deserialized double matrix
     * @throws IOException
     *             if anything goes wrong (e.g., there is no double matrix
     *             stored in that file)
     */
    public static MatrixD deserializeD(Path file) throws IOException {
        //@formatter:off
        try (InputStream is = Files.newInputStream(file);
             BufferedInputStream bis = new BufferedInputStream(is, BUF_SIZE)
        )
        {
            return deserializeD(bis);
        }
        //@formatter:on
    }

    /**
     * Attempts to read a complex double matrix from the provided Path
     * {@code file}.
     * 
     * @param file
     *            the file to read the complex double matrix from
     * @return the deserialized complex double matrix
     * @throws IOException
     *             if anything goes wrong (e.g., there is no complex double
     *             matrix stored in that file)
     */
    public static ComplexMatrixD deserializeComplexD(Path file) throws IOException {
        //@formatter:off
        try (InputStream is = Files.newInputStream(file);
             BufferedInputStream bis = new BufferedInputStream(is, BUF_SIZE)
        )
        {
            return deserializeComplexD(bis);
        }
        //@formatter:on
    }

    /**
     * Attempts to read a double matrix from the current position of the
     * provided input stream {@code is}. The correct positioning of the input
     * stream and closing the input stream must be done by the caller.
     * 
     * @param is
     *            the input stream to read a double matrix from
     * @return the double matrix deserialized from the input stream
     * @throws IOException
     *             if anything goes wrong (e.g., the position is wrong or there
     *             is no double matrix stored at the current position)
     */
    public static MatrixD deserializeD(InputStream is) throws IOException {
        byte[] buf = new byte[8];
        checkBigendian(IO.isBigendian(buf, is));
        if (!IO.isDoubleType(buf, is)) {
            throw new IOException("Unexpected float type. Use float Deserializer instead.");
        }
        if (IO.isComplexType(buf)) {
            throw new IOException("Unexpected ComplexMatrixD. Use deserializeComplexD() instead.");
        }
        int rows = IO.readRows(true, buf, is);
        int cols = IO.readCols(true, buf, is);
        MatrixD md = createD(rows, cols);
        double[] data = md.getArrayUnsafe();
        for (int i = 0; i < data.length; ++i) {
            data[i] = IO.getDoubleB(buf, is);
        }
        return md;
    }

    /**
     * Attempts to read a complex double matrix from the current position of the
     * provided input stream {@code is}. The correct positioning of the input
     * stream and closing the input stream must be done by the caller.
     * 
     * @param is
     *            the input stream to read a complex double matrix from
     * @return the complex double matrix deserialized from the input stream
     * @throws IOException
     *             if anything goes wrong (e.g., the position is wrong or there
     *             is no complex double matrix stored at the current position)
     */
    public static ComplexMatrixD deserializeComplexD(InputStream is) throws IOException {
        byte[] buf = new byte[8];
        checkBigendian(IO.isBigendian(buf, is));
        if (!IO.isDoubleType(buf, is)) {
            throw new IOException("Unexpected float type. Use float Deserializer instead.");
        }
        if (!IO.isComplexType(buf)) {
            throw new IOException("Unexpected MatrixD. Use deserializeD() instead.");
        }
        int rows = IO.readRows(true, buf, is);
        int cols = IO.readCols(true, buf, is);
        ComplexMatrixD cmd = createComplexD(rows, cols);
        double[] data = cmd.getArrayUnsafe();
        for (int i = 0; i < data.length; ++i) {
            data[i] = IO.getDoubleB(buf, is);
        }
        return cmd;
    }

    /**
     * Create a {@code MatrixD} copy of the {@code MatrixF} input matrix.
     * 
     * @param mf
     *            {@code MatrixF} input matrix to convert
     * @return a {@code MatrixD} copy of the input matrix
     */
    public static MatrixD convert(MatrixF mf) {
        MatrixD md = createD(mf.numRows(), mf.numColumns());
        double[] ad = md.getArrayUnsafe();
        float[] fd = mf.getArrayUnsafe();
        for (int i = 0; i < ad.length; ++i) {
            ad[i] = fd[i];
        }
        return md;
    }

    /**
     * Create a {@code MatrixF} copy of the {@code MatrixD} input matrix.
     * 
     * @param md
     *            {@code MatrixD} input matrix to convert
     * @return a {@code MatrixF} copy of the input matrix
     */
    public static MatrixF convert(MatrixD md) {
        MatrixF mf = createF(md.numRows(), md.numColumns());
        float[] fd = mf.getArrayUnsafe();
        double[] ad = md.getArrayUnsafe();
        for (int i = 0; i < fd.length; ++i) {
            fd[i] = (float) ad[i];
        }
        return mf;
    }

    /**
     * Create a {@code ComplexMatrixD} copy of the {@code ComplexMatrixF} input
     * matrix.
     * 
     * @param cmf
     *            {@code ComplexMatrixF} input matrix to convert
     * @return a {@code ComlexMatrixD} copy of the input matrix
     */
    public static ComplexMatrixD convert(ComplexMatrixF cmf) {
        ComplexMatrixD cmd = createComplexD(cmf.numRows(), cmf.numColumns());
        double[] ad = cmd.getArrayUnsafe();
        float[] fd = cmf.getArrayUnsafe();
        for (int i = 0; i < ad.length; ++i) {
            ad[i] = fd[i];
        }
        return cmd;
    }

    /**
     * Create a {@code ComplexMatrixF} copy of the {@code ComplexMatrixD} input
     * matrix.
     * 
     * @param cmd
     *            {@code CompleyMatrixD} input matrix to convert
     * @return a {@code ComplexMatrixF} copy of the input matrix
     */
    public static ComplexMatrixF convert(ComplexMatrixD cmd) {
        ComplexMatrixF cmf = createComplexF(cmd.numRows(), cmd.numColumns());
        float[] fd = cmf.getArrayUnsafe();
        double[] ad = cmd.getArrayUnsafe();
        for (int i = 0; i < fd.length; ++i) {
            fd[i] = (float) ad[i];
        }
        return cmf;
    }

    /**
     * Create a {@code ComplexMatrixF} copy of the {@code MatrixF} input matrix
     * where the real parts are copied from {@code mf} and the imaginary parts
     * are set to {@code 0.0f}.
     * 
     * @param mf
     *            {@code MatrixF} input matrix to copy into a
     *            {@code ComplexMatrixF}
     * @return a {@code ComplexMatrixF} copy of the input matrix
     */
    public static ComplexMatrixF convertToComplex(MatrixF mf) {
        ComplexMatrixF cmf = createComplexF(mf.numRows(), mf.numColumns());
        float[] to = cmf.getArrayUnsafe();
        float[] from = mf.getArrayUnsafe();
        for (int i = 0; i < from.length; ++i) {
            to[2 * i] = from[i];
        }
        return cmf;
    }

    /**
     * Create a {@code ComplexMatrixD} copy of the {@code MatrixD} input matrix
     * where the real parts are copied from {@code md} and the imaginary parts
     * are set to {@code 0.0}.
     * 
     * @param md
     *            {@code MatrixD} input matrix to copy into a
     *            {@code ComplexMatrixD}
     * @return a {@code ComplexMatrixD} copy of the input matrix
     */
    public static ComplexMatrixD convertToComplex(MatrixD md) {
        ComplexMatrixD cmd = createComplexD(md.numRows(), md.numColumns());
        double[] to = cmd.getArrayUnsafe();
        double[] from = md.getArrayUnsafe();
        for (int i = 0; i < from.length; ++i) {
            to[2 * i] = from[i];
        }
        return cmd;
    }

    /**
     * Create a {@code MatrixF} copy of the {@code ComplexMatrixF} input matrix
     * where only the real parts are copied from {@code cmf}.
     * 
     * @param cmf
     *            {@code ComplexMatrixF} input matrix to copy into a
     *            {@code MatrixF}
     * @return a {@code MatrixF} copy of the real parts of the input matrix
     */
    public static MatrixF convertToReal(ComplexMatrixF cmf) {
        MatrixF mf = createF(cmf.numRows(), cmf.numColumns());
        float[] to = mf.getArrayUnsafe();
        float[] from = cmf.getArrayUnsafe();
        for (int i = 0; i < to.length; ++i) {
            to[i] = from[2 * i];
        }
        return mf;
    }

    /**
     * Create a {@code MatrixD} copy of the {@code ComplexMatrixD} input matrix
     * where only the real parts are copied from {@code cmd}.
     * 
     * @param cmd
     *            {@code ComplexMatrixD} input matrix to copy into a
     *            {@code MatrixD}
     * @return a {@code MatrixD} copy of the real parts of the input matrix
     */
    public static MatrixD convertToReal(ComplexMatrixD cmd) {
        MatrixD md = createD(cmd.numRows(), cmd.numColumns());
        double[] to = md.getArrayUnsafe();
        double[] from = cmd.getArrayUnsafe();
        for (int i = 0; i < to.length; ++i) {
            to[i] = from[2 * i];
        }
        return md;
    }

    /**
     * Create a {@code column.length x 1} column vector from the provided array
     * which must have at least length {@code 1} and contains the entries of the
     * new column vector.
     * 
     * @param column
     *            array containing the values of the elements of the vector to
     *            be created. Must not be {@code null} and must have length
     *            {@code > 0}.
     * @return a {@code column.length x 1} MatrixD column vector
     */
    public static MatrixD colVectorD(double[] column) {
        if (Objects.requireNonNull(column).length == 0) {
            throw new IllegalArgumentException("column array length must be > 0");
        }
        return new SimpleMatrixD(column.length, 1, Arrays.copyOf(column, column.length));
    }

    /**
     * Create a {@code column.length x 1} column vector from the provided array
     * which must have at least length {@code 1} and contains the entries of the
     * new column vector.
     * 
     * @param column
     *            array containing the values of the elements of the vector to
     *            be created. Must not be {@code null} and must have length
     *            {@code > 0}.
     * @return a {@code column.length x 1} MatrixF column vector
     */
    public static MatrixF colVectorF(float[] column) {
        if (Objects.requireNonNull(column).length == 0) {
            throw new IllegalArgumentException("column array length must be > 0");
        }
        return new SimpleMatrixF(column.length, 1, Arrays.copyOf(column, column.length));
    }

    /**
     * Create a {@code column.length x 1} column vector from the provided array
     * which must have at least length {@code 1} and contains the entries of the
     * new column vector.
     * 
     * @param column
     *            array containing the values of the elements of the vector to
     *            be created. Must not be {@code null} and must have length
     *            {@code > 0}.
     * @return a {@code column.length x 1} ComplexMatrixD column vector
     */
    public static ComplexMatrixD colVectorComplexD(Zd[] column) {
        if (Objects.requireNonNull(column).length == 0) {
            throw new IllegalArgumentException("column array length must be > 0");
        }
        return new SimpleComplexMatrixD(column.length, 1, ZArrayUtil.complexToPrimitiveArray(column));
    }

    /**
     * Create a {@code column.length x 1} column vector from the provided array
     * which must have at least length {@code 1} and contains the entries of the
     * new column vector.
     * 
     * @param column
     *            array containing the values of the elements of the vector to
     *            be created. Must not be {@code null} and must have length
     *            {@code > 0}.
     * @return a {@code column.length x 1} ComplexMatrixF column vector
     */
    public static ComplexMatrixF colVectorComplexF(Zf[] column) {
        if (Objects.requireNonNull(column).length == 0) {
            throw new IllegalArgumentException("column array length must be > 0");
        }
        return new SimpleComplexMatrixF(column.length, 1, ZArrayUtil.complexToPrimitiveArray(column));
    }

    /**
     * Returns the distance between {@code A} and {@code B} computed as the
     * {@code 1-norm} for vectors (a.k.a. {@code L1} norm, Manhattan distance or
     * taxicab norm for vectors) of {@code A - B}, i.e., the sum of the absolute
     * values of the differences between the matrix entries. This computation is
     * symmetric, so interchanging {@code A} and {@code B} doesn't change the
     * result.
     * 
     * @param A
     *            the first matrix to use for the distance computation (it
     *            doesn't matter which one since {@code L1} is a distance in the
     *            metric space sense)
     * @param B
     *            the other matrix to use for the distance computation (it
     *            doesn't matter which one since {@code L1} is a distance in the
     *            metric space sense)
     * @return the "entrywise" <code>L<sub>1,1</sub></code> norm of
     *         {@code A - B}
     * @throws IndexOutOfBoundsException
     *             if {@code A} and {@code B} do not have the same dimension
     */
    public static double distance(MatrixD A, MatrixD B) {// vectorization prospect
        if (A == B) {
            return 0.0;
        }
        Checks.checkEqualDimension(A, B);
        double[] _a = A.getArrayUnsafe();
        double[] _b = B.getArrayUnsafe();
        double d1 = 0.0;
        for (int i = 0; i < _a.length; ++i) {
            double a = _a[i];
            double b = _b[i];
            if (a != b) {
                d1 += Math.abs(a - b);
            }
        }
        return d1;
    }

    /**
     * Returns the distance between {@code A} and {@code B} computed as the
     * {@code 1-norm} for vectors (a.k.a. {@code L1} norm, Manhattan distance or
     * taxicab norm for vectors) of {@code A - B}, i.e., the sum of the absolute
     * values of the differences between the matrix entries. This computation is
     * symmetric, so interchanging {@code A} and {@code B} doesn't change the
     * result.
     * 
     * @param A
     *            the first matrix to use for the distance computation (it
     *            doesn't matter which one since {@code L1} is a distance in the
     *            metric space sense)
     * @param B
     *            the other matrix to use for the distance computation (it
     *            doesn't matter which one since {@code L1} is a distance in the
     *            metric space sense)
     * @return the "entrywise" <code>L<sub>1,1</sub></code> norm of
     *         {@code A - B}
     * @throws IndexOutOfBoundsException
     *             if {@code A} and {@code B} do not have the same dimension
     */
    public static float distance(MatrixF A, MatrixF B) {
        if (A == B) {
            return 0.0f;
        }
        Checks.checkEqualDimension(A, B);
        float[] _a = A.getArrayUnsafe();
        float[] _b = B.getArrayUnsafe();
        double d1 = 0.0;
        for (int i = 0; i < _a.length; ++i) {
            float a = _a[i];
            float b = _b[i];
            if (a != b) {
                d1 += Math.abs(a - b);
            }
        }
        return (float) d1;
    }

    /**
     * Returns the distance between {@code A} and {@code B} computed as the
     * {@code 1-norm} for vectors (a.k.a. {@code L1} norm, Manhattan distance or
     * taxicab norm for vectors) of {@code A - B}, i.e., the sum of the absolute
     * values of the differences between the matrix entries. This computation is
     * symmetric, so interchanging {@code A} and {@code B} doesn't change the
     * result.
     * 
     * @param A
     *            the first matrix to use for the distance computation (it
     *            doesn't matter which one since {@code L1} is a distance in the
     *            metric space sense)
     * @param B
     *            the other matrix to use for the distance computation (it
     *            doesn't matter which one since {@code L1} is a distance in the
     *            metric space sense)
     * @return the "entrywise" <code>L<sub>1,1</sub></code> norm of
     *         {@code A - B}
     * @throws IndexOutOfBoundsException
     *             if {@code A} and {@code B} do not have the same dimension
     */
    public static double distance(ComplexMatrixD A, ComplexMatrixD B) {
        if (A == B) {
            return 0.0;
        }
        Checks.checkEqualDimension(A, B);
        double[] _a = A.getArrayUnsafe();
        double[] _b = B.getArrayUnsafe();
        double d1 = 0.0;
        for (int i = 0; i < _a.length; i += 2) {
            double rea = _a[i];
            double ima = _a[i + 1]; // "lgtm[java/index-out-of-bounds]"
            double reb = _b[i];
            double imb = _b[i + 1];
            if (rea != reb || ima != imb) {
                d1 += ZdImpl.abs(rea - reb, ima - imb);
            }
        }
        return d1;
    }

    /**
     * Returns the distance between {@code A} and {@code B} computed as the
     * {@code 1-norm} for vectors (a.k.a. {@code L1} norm, Manhattan distance or
     * taxicab norm for vectors) of {@code A - B}, i.e., the sum of the absolute
     * values of the differences between the matrix entries. This computation is
     * symmetric, so interchanging {@code A} and {@code B} doesn't change the
     * result.
     * 
     * @param A
     *            the first matrix to use for the distance computation (it
     *            doesn't matter which one since {@code L1} is a distance in the
     *            metric space sense)
     * @param B
     *            the other matrix to use for the distance computation (it
     *            doesn't matter which one since {@code L1} is a distance in the
     *            metric space sense)
     * @return the "entrywise" <code>L<sub>1,1</sub></code> norm of
     *         {@code A - B}
     * @throws IndexOutOfBoundsException
     *             if {@code A} and {@code B} do not have the same dimension
     */
    public static float distance(ComplexMatrixF A, ComplexMatrixF B) {
        if (A == B) {
            return 0.0f;
        }
        Checks.checkEqualDimension(A, B);
        float[] _a = A.getArrayUnsafe();
        float[] _b = B.getArrayUnsafe();
        double d1 = 0.0;
        for (int i = 0; i < _a.length; i += 2) {
            // use higher precision internally
            double rea = _a[i];
            double ima = _a[i + 1]; // "lgtm[java/index-out-of-bounds]"
            double reb = _b[i];
            double imb = _b[i + 1]; // "lgtm[java/index-out-of-bounds]"
            if (rea != reb || ima != imb) {
                d1 += ZdImpl.abs(rea - reb, ima - imb);
            }
        }
        return (float) d1;
    }

    /**
     * Tests for approximate equality (or "closeness") of the two matrices
     * {@code A} and {@code B} where {@code A} and {@code B} must have the same
     * dimension and each element {@code a} of {@code A} is tested for
     * approximate equality against the corresponding element {@code b} of
     * {@code B} as defined in
     * {@link #approxEqual(MatrixD, MatrixD, double, double)}. This method is
     * equivalent to a call of
     * {@linkplain #approxEqual(MatrixD, MatrixD, double, double)} with an
     * {@code relTol} argument of value {@code 1.0e-8} and an {@code absTol}
     * argument equal to {@code 0.0}:
     * 
     * <pre>
     * {@code approxEqual(A, B, 1.0e-8, 0.0)}
     * </pre>
     * 
     * @param A
     *            one of the two matrices to test for approximate equality (it
     *            doesn't matter which one since the test is symmetric)
     * @param B
     *            the other one of the two matrices to test for approximate
     *            equality (it doesn't matter which one since the test is
     *            symmetric)
     * @return {@code true} if {@code A} and {@code B} are approximately equal
     *         according to the criterion defined in
     *         {@link #approxEqual(MatrixD, MatrixD, double, double)}
     */
    public static boolean approxEqual(MatrixD A, MatrixD B) {
        return approxEqual(A, B, 1.0e-8);
    }

    /**
     * Tests for approximate equality (or "closeness") of the two matrices
     * {@code A} and {@code B} where {@code A} and {@code B} must have the same
     * dimension and each element {@code a} of {@code A} is tested for
     * approximate equality against the corresponding element {@code b} of
     * {@code B} as defined in
     * {@link #approxEqual(MatrixD, MatrixD, double, double)}. This method is
     * equivalent to a call of
     * {@linkplain #approxEqual(MatrixD, MatrixD, double, double)} with an
     * {@code absTol} argument equal to {@code 0.0}:
     * 
     * <pre>
     * {@code approxEqual(A, B, relTol, 0.0)}
     * </pre>
     * 
     * @param A
     *            one of the two matrices to test for approximate equality (it
     *            doesn't matter which one since the test is symmetric)
     * @param B
     *            the other one of the two matrices to test for approximate
     *            equality (it doesn't matter which one since the test is
     *            symmetric)
     * @param relTol
     *            relative tolerance, must be {@code >= 0.0}
     * @return {@code true} if {@code A} and {@code B} are approximately equal
     *         according to the criterion defined in
     *         {@link #approxEqual(MatrixD, MatrixD, double, double)}
     * @throws IllegalArgumentException
     *             if {@code relTol < 0.0}
     */
    public static boolean approxEqual(MatrixD A, MatrixD B, double relTol) {
        return approxEqual(A, B, relTol, 0.0);
    }

    /**
     * Tests for approximate equality (or "closeness") of the two matrices
     * {@code A} and {@code B} where {@code A} and {@code B} must have the same
     * dimension and each element {@code a} of {@code A} is tested for
     * approximate equality against the corresponding element {@code b} of
     * {@code B}.
     * <p>
     * If, for all pairs {@code (a, b)},
     * 
     * <pre>
     * {@code abs(a - b) <= max( relTol * max(abs(a), abs(b)), absTol )}
     * </pre>
     * 
     * the matrices {@code A} and {@code B} are considered approximately equal,
     * otherwise they are not. This test is symmetric, so interchanging
     * {@code A} and {@code B} doesn't change the result.
     * <p>
     * <b>Implementation Note:</b><br>
     * The definition of approximate equality used here is the one employed in
     * Python's {@code math.isclose()} function defined in
     * <a href=https://www.python.org/dev/peps/pep-0485/>PEP 485 - A Function
     * for testing approximate equality</a>. This document gives a nice
     * discussion of the rationale for this approach, how to use it, and the
     * alternatives they had considered.
     * 
     * @param A
     *            one of the two matrices to test for approximate equality (it
     *            doesn't matter which one since the test is symmetric)
     * @param B
     *            the other one of the two matrices to test for approximate
     *            equality (it doesn't matter which one since the test is
     *            symmetric)
     * @param relTol
     *            relative tolerance, must be {@code >= 0.0}
     * @param absTol
     *            absolute tolerance, must be {@code >= 0.0}
     * @return {@code true} if {@code A} and {@code B} are approximately equal
     *         according to the criterion defined above, otherwise {@code false}
     * @throws IllegalArgumentException
     *             if {@code relTol < 0.0} or {@code absTol < 0.0}
     */
    public static boolean approxEqual(MatrixD A, MatrixD B, double relTol, double absTol) {
        if (!checkApproxEqualArgs(A, B, relTol, absTol)) {// vectorization prospect
            return false;
        }
        if (A == B) {
            return true;
        }
        double[] _a = A.getArrayUnsafe();
        double[] _b = B.getArrayUnsafe();
        for (int i = 0; i < _a.length; ++i) {
            double a = _a[i];
            double b = _b[i];
            if (a != b) {
                double diff = Math.abs(a - b);
                if (!((diff <= relTol * Math.max(Math.abs(a), Math.abs(b))) || (diff <= absTol))) {
                    return false;
                }
            }
        }
        return true;
    }

    /**
     * Tests for approximate equality (or "closeness") of the two matrices
     * {@code A} and {@code B} where {@code A} and {@code B} must have the same
     * dimension and each element {@code a} of {@code A} is tested for
     * approximate equality against the corresponding element {@code b} of
     * {@code B} as defined in
     * {@link #approxEqual(MatrixF, MatrixF, float, float)}. This method is
     * equivalent to a call of
     * {@linkplain #approxEqual(MatrixF, MatrixF, float, float)} with an
     * {@code relTol} argument of value {@code 1.0e-4f} and an {@code absTol}
     * argument equal to {@code 0.0f}:
     * 
     * <pre>
     * {@code approxEqual(A, B, 1.0e-4f, 0.0f)}
     * </pre>
     * 
     * @param A
     *            one of the two matrices to test for approximate equality (it
     *            doesn't matter which one since the test is symmetric)
     * @param B
     *            the other one of the two matrices to test for approximate
     *            equality (it doesn't matter which one since the test is
     *            symmetric)
     * @return {@code true} if {@code A} and {@code B} are approximately equal
     *         according to the criterion defined in
     *         {@link #approxEqual(MatrixF, MatrixF, float, float)}
     */
    public static boolean approxEqual(MatrixF A, MatrixF B) {
        return approxEqual(A, B, 1.0e-4f);
    }

    /**
     * Tests for approximate equality (or "closeness") of the two matrices
     * {@code A} and {@code B} where {@code A} and {@code B} must have the same
     * dimension and each element {@code a} of {@code A} is tested for
     * approximate equality against the corresponding element {@code b} of
     * {@code B} as defined in
     * {@link #approxEqual(MatrixF, MatrixF, float, float)}. This method is
     * equivalent to a call of
     * {@linkplain #approxEqual(MatrixF, MatrixF, float, float)} with an
     * {@code absTol} argument equal to {@code 0.0f}:
     * 
     * <pre>
     * {@code approxEqual(A, B, relTol, 0.0f)}
     * </pre>
     * 
     * @param A
     *            one of the two matrices to test for approximate equality (it
     *            doesn't matter which one since the test is symmetric)
     * @param B
     *            the other one of the two matrices to test for approximate
     *            equality (it doesn't matter which one since the test is
     *            symmetric)
     * @param relTol
     *            relative tolerance, must be {@code >= 0.0f}
     * @return {@code true} if {@code A} and {@code B} are approximately equal
     *         according to the criterion defined in
     *         {@link #approxEqual(MatrixF, MatrixF, float, float)}
     * @throws IllegalArgumentException
     *             if {@code relTol < 0.0f}
     */
    public static boolean approxEqual(MatrixF A, MatrixF B, float relTol) {
        return approxEqual(A, B, relTol, 0.0f);
    }

    /**
     * Tests for approximate equality (or "closeness") of the two matrices
     * {@code A} and {@code B} where {@code A} and {@code B} must have the same
     * dimension and each element {@code a} of {@code A} is tested for
     * approximate equality against the corresponding element {@code b} of
     * {@code B}.
     * <p>
     * If, for all pairs {@code (a, b)},
     * 
     * <pre>
     * {@code abs(a - b) <= max( relTol * max(abs(a), abs(b)), absTol )}
     * </pre>
     * 
     * the matrices {@code A} and {@code B} are considered approximately equal,
     * otherwise they are not. This test is symmetric, so interchanging
     * {@code A} and {@code B} doesn't change the result.
     * <p>
     * <b>Implementation Note:</b><br>
     * The definition of approximate equality used here is the one employed in
     * Python's {@code math.isclose()} function defined in
     * <a href=https://www.python.org/dev/peps/pep-0485/>PEP 485 - A Function
     * for testing approximate equality</a>. This document gives a nice
     * discussion of the rationale for this approach, how to use it, and the
     * alternatives they had considered.
     * 
     * @param A
     *            one of the two matrices to test for approximate equality (it
     *            doesn't matter which one since the test is symmetric)
     * @param B
     *            the other one of the two matrices to test for approximate
     *            equality (it doesn't matter which one since the test is
     *            symmetric)
     * @param relTol
     *            relative tolerance, must be {@code >= 0.0f}
     * @param absTol
     *            absolute tolerance, must be {@code >= 0.0f}
     * @return {@code true} if {@code A} and {@code B} are approximately equal
     *         according to the criterion defined above, otherwise {@code false}
     * @throws IllegalArgumentException
     *             if {@code relTol < 0.0f} or {@code absTol < 0.0f}
     */
    public static boolean approxEqual(MatrixF A, MatrixF B, float relTol, float absTol) {
        if (!checkApproxEqualArgs(A, B, relTol, absTol)) {
            return false;
        }
        if (A == B) {
            return true;
        }
        float[] _a = A.getArrayUnsafe();
        float[] _b = B.getArrayUnsafe();
        for (int i = 0; i < _a.length; ++i) {
            float a = _a[i];
            float b = _b[i];
            if (a != b) {
                float diff = Math.abs(a - b);
                if (!((diff <= relTol * Math.max(Math.abs(a), Math.abs(b))) || (diff <= absTol))) {
                    return false;
                }
            }
        }
        return true;
    }

    /**
     * Tests for approximate equality (or "closeness") of the two matrices
     * {@code A} and {@code B} where {@code A} and {@code B} must have the same
     * dimension and each element {@code a} of {@code A} is tested for
     * approximate equality against the corresponding element {@code b} of
     * {@code B} as defined in
     * {@link #approxEqual(ComplexMatrixD, ComplexMatrixD, double, double)}.
     * This method is equivalent to a call of
     * {@linkplain #approxEqual(ComplexMatrixD, ComplexMatrixD, double, double)}
     * with an {@code relTol} argument of value {@code 1.0e-8} and an
     * {@code absTol} argument equal to {@code 0.0}:
     * 
     * <pre>
     * {@code approxEqual(A, B, 1.0e-8, 0.0)}
     * </pre>
     * 
     * @param A
     *            one of the two matrices to test for approximate equality (it
     *            doesn't matter which one since the test is symmetric)
     * @param B
     *            the other one of the two matrices to test for approximate
     *            equality (it doesn't matter which one since the test is
     *            symmetric)
     * @return {@code true} if {@code A} and {@code B} are approximately equal
     *         according to the criterion defined in
     *         {@link #approxEqual(ComplexMatrixD, ComplexMatrixD, double, double)}
     */
    public static boolean approxEqual(ComplexMatrixD A, ComplexMatrixD B) {
        return approxEqual(A, B, 1.0e-8);
    }

    /**
     * Tests for approximate equality (or "closeness") of the two matrices
     * {@code A} and {@code B} where {@code A} and {@code B} must have the same
     * dimension and each element {@code a} of {@code A} is tested for
     * approximate equality against the corresponding element {@code b} of
     * {@code B} as defined in
     * {@link #approxEqual(ComplexMatrixD, ComplexMatrixD, double, double)}.
     * This method is equivalent to a call of
     * {@linkplain #approxEqual(ComplexMatrixD, ComplexMatrixD, double, double)}
     * with an {@code absTol} argument equal to {@code 0.0}:
     * 
     * <pre>
     * {@code approxEqual(A, B, relTol, 0.0)}
     * </pre>
     * 
     * @param A
     *            one of the two matrices to test for approximate equality (it
     *            doesn't matter which one since the test is symmetric)
     * @param B
     *            the other one of the two matrices to test for approximate
     *            equality (it doesn't matter which one since the test is
     *            symmetric)
     * @param relTol
     *            relative tolerance, must be {@code >= 0.0}
     * @return {@code true} if {@code A} and {@code B} are approximately equal
     *         according to the criterion defined in
     *         {@link #approxEqual(ComplexMatrixD, ComplexMatrixD, double, double)}
     * @throws IllegalArgumentException
     *             if {@code relTol < 0.0}
     */
    public static boolean approxEqual(ComplexMatrixD A, ComplexMatrixD B, double relTol) {
        return approxEqual(A, B, relTol, 0.0);
    }

    /**
     * Tests for approximate equality (or "closeness") of the two matrices
     * {@code A} and {@code B} where {@code A} and {@code B} must have the same
     * dimension and each element {@code a} of {@code A} is tested for
     * approximate equality against the corresponding element {@code b} of
     * {@code B}.
     * <p>
     * If, for all pairs {@code (a, b)},
     * 
     * <pre>
     * {@code abs(a - b) <= max( relTol * max(abs(a), abs(b)), absTol )}
     * </pre>
     * 
     * the matrices {@code A} and {@code B} are considered approximately equal,
     * otherwise they are not. This test is symmetric, so interchanging
     * {@code A} and {@code B} doesn't change the result.
     * <p>
     * <b>Implementation Note:</b><br>
     * The definition of approximate equality used here is the one employed in
     * Python's {@code cmath.isclose()} function defined in
     * <a href=https://www.python.org/dev/peps/pep-0485/>PEP 485 - A Function
     * for testing approximate equality</a>. This document gives a nice
     * discussion of the rationale for this approach, how to use it, and the
     * alternatives they had considered.
     * 
     * @param A
     *            one of the two matrices to test for approximate equality (it
     *            doesn't matter which one since the test is symmetric)
     * @param B
     *            the other one of the two matrices to test for approximate
     *            equality (it doesn't matter which one since the test is
     *            symmetric)
     * @param relTol
     *            relative tolerance, must be {@code >= 0.0}
     * @param absTol
     *            absolute tolerance, must be {@code >= 0.0}
     * @return {@code true} if {@code A} and {@code B} are approximately equal
     *         according to the criterion defined above, otherwise {@code false}
     * @throws IllegalArgumentException
     *             if {@code relTol < 0.0} or {@code absTol < 0.0}
     */
    public static boolean approxEqual(ComplexMatrixD A, ComplexMatrixD B, double relTol, double absTol) {
        if (!checkApproxEqualArgs(A, B, relTol, absTol)) {
            return false;
        }
        if (A == B) {
            return true;
        }
        double[] _a = A.getArrayUnsafe();
        double[] _b = B.getArrayUnsafe();
        for (int i = 0; i < _a.length; i += 2) {
            double a_re = _a[i];
            double a_im = _a[i + 1]; // "lgtm[java/index-out-of-bounds]"
            double b_re = _b[i];
            double b_im = _b[i + 1]; // "lgtm[java/index-out-of-bounds]"
            if (a_re != b_re || a_im != b_im) {
                double diff = ZdImpl.abs(a_re - b_re, a_im - b_im);
                if (!((diff <= relTol * Math.max(ZdImpl.abs(a_re, a_im), ZdImpl.abs(b_re, b_im)))
                        || (diff <= absTol))) {
                    return false;
                }
            }
        }
        return true;
    }

    /**
     * Tests for approximate equality (or "closeness") of the two matrices
     * {@code A} and {@code B} where {@code A} and {@code B} must have the same
     * dimension and each element {@code a} of {@code A} is tested for
     * approximate equality against the corresponding element {@code b} of
     * {@code B} as defined in
     * {@link #approxEqual(ComplexMatrixF, ComplexMatrixF, float, float)}. This
     * method is equivalent to a call of
     * {@linkplain #approxEqual(ComplexMatrixF, ComplexMatrixF, float, float)}
     * with an {@code relTol} argument of value {@code 1.0e-4f} and an
     * {@code absTol} argument equal to {@code 0.0f}:
     * 
     * <pre>
     * {@code approxEqual(A, B, 1.0e-4f, 0.0f)}
     * </pre>
     * 
     * @param A
     *            one of the two matrices to test for approximate equality (it
     *            doesn't matter which one since the test is symmetric)
     * @param B
     *            the other one of the two matrices to test for approximate
     *            equality (it doesn't matter which one since the test is
     *            symmetric)
     * @return {@code true} if {@code A} and {@code B} are approximately equal
     *         according to the criterion defined in
     *         {@link #approxEqual(ComplexMatrixF, ComplexMatrixF, float, float)}
     */
    public static boolean approxEqual(ComplexMatrixF A, ComplexMatrixF B) {
        return approxEqual(A, B, 1.0e-4f);
    }

    /**
     * Tests for approximate equality (or "closeness") of the two matrices
     * {@code A} and {@code B} where {@code A} and {@code B} must have the same
     * dimension and each element {@code a} of {@code A} is tested for
     * approximate equality against the corresponding element {@code b} of
     * {@code B} as defined in
     * {@link #approxEqual(ComplexMatrixF, ComplexMatrixF, float, float)}. This
     * method is equivalent to a call of
     * {@linkplain #approxEqual(ComplexMatrixF, ComplexMatrixF, float, float)}
     * with an {@code absTol} argument equal to {@code 0.0f}:
     * 
     * <pre>
     * {@code approxEqual(A, B, relTol, 0.0f)}
     * </pre>
     * 
     * @param A
     *            one of the two matrices to test for approximate equality (it
     *            doesn't matter which one since the test is symmetric)
     * @param B
     *            the other one of the two matrices to test for approximate
     *            equality (it doesn't matter which one since the test is
     *            symmetric)
     * @param relTol
     *            relative tolerance, must be {@code >= 0.0f}
     * @return {@code true} if {@code A} and {@code B} are approximately equal
     *         according to the criterion defined in
     *         {@link #approxEqual(ComplexMatrixF, ComplexMatrixF, float, float)}
     * @throws IllegalArgumentException
     *             if {@code relTol < 0.0f}
     */
    public static boolean approxEqual(ComplexMatrixF A, ComplexMatrixF B, float relTol) {
        return approxEqual(A, B, relTol, 0.0f);
    }

    /**
     * Tests for approximate equality (or "closeness") of the two matrices
     * {@code A} and {@code B} where {@code A} and {@code B} must have the same
     * dimension and each element {@code a} of {@code A} is tested for
     * approximate equality against the corresponding element {@code b} of
     * {@code B}.
     * <p>
     * If, for all pairs {@code (a, b)},
     * 
     * <pre>
     * {@code abs(a - b) <= max( relTol * max(abs(a), abs(b)), absTol )}
     * </pre>
     * 
     * the matrices {@code A} and {@code B} are considered approximately equal,
     * otherwise they are not. This test is symmetric, so interchanging
     * {@code A} and {@code B} doesn't change the result.
     * <p>
     * <b>Implementation Note:</b><br>
     * The definition of approximate equality used here is the one employed in
     * Python's {@code cmath.isclose()} function defined in
     * <a href=https://www.python.org/dev/peps/pep-0485/>PEP 485 - A Function
     * for testing approximate equality</a>. This document gives a nice
     * discussion of the rationale for this approach, how to use it, and the
     * alternatives they had considered.
     * 
     * @param A
     *            one of the two matrices to test for approximate equality (it
     *            doesn't matter which one since the test is symmetric)
     * @param B
     *            the other one of the two matrices to test for approximate
     *            equality (it doesn't matter which one since the test is
     *            symmetric)
     * @param relTol
     *            relative tolerance, must be {@code >= 0.0f}
     * @param absTol
     *            absolute tolerance, must be {@code >= 0.0f}
     * @return {@code true} if {@code A} and {@code B} are approximately equal
     *         according to the criterion defined above, otherwise {@code false}
     * @throws IllegalArgumentException
     *             if {@code relTol < 0.0f} or {@code absTol < 0.0f}
     */
    public static boolean approxEqual(ComplexMatrixF A, ComplexMatrixF B, float relTol, float absTol) {
        if (!checkApproxEqualArgs(A, B, relTol, absTol)) {
            return false;
        }
        if (A == B) {
            return true;
        }
        float[] _a = A.getArrayUnsafe();
        float[] _b = B.getArrayUnsafe();
        for (int i = 0; i < _a.length; i += 2) {
            float a_re = _a[i];
            float a_im = _a[i + 1]; // "lgtm[java/index-out-of-bounds]"
            float b_re = _b[i];
            float b_im = _b[i + 1]; // "lgtm[java/index-out-of-bounds]"
            if (a_re != b_re || a_im != b_im) {
                double diff = ZfImpl.abs(a_re - b_re, a_im - b_im);
                if (!((diff <= relTol * Math.max(ZfImpl.abs(a_re, a_im), ZfImpl.abs(b_re, b_im)))
                        || (diff <= absTol))) {
                    return false;
                }
            }
        }
        return true;
    }

    private static boolean checkApproxEqualArgs(Dimensions A, Dimensions B, double relTol, double absTol) {
        if (relTol < 0.0) {
            throw new IllegalArgumentException("relTol < 0.0 : " + relTol);
        }
        if (absTol < 0.0) {
            throw new IllegalArgumentException("absTol < 0.0 : " + absTol);
        }
        if (A.numRows() != B.numRows() || A.numColumns() != B.numColumns()) {
            return false;
        }
        return true;
    }

    /* package */ static String toString(Dimensions dim) {
        StringBuilder buf = new StringBuilder();
        buf.append(dim.asString()).append(System.lineSeparator());
        int _cols = dim.numColumns() <= MAX_ROWCOL ? dim.numColumns() : LAST_IDX;
        int _rows = dim.numRows() <= MAX_ROWCOL ? dim.numRows() : LAST_IDX;
        int row;
        for (row = 0; row < _rows; ++row) {
            if (dim instanceof MatrixD) {
                printRowD(row, _cols, (MatrixD) dim, buf);
            } else if (dim instanceof MatrixF) {
                printRowF(row, _cols, (MatrixF) dim, buf);
            } else if (dim instanceof ComplexMatrixD) {
                printRowComplexD(row, _cols, (ComplexMatrixD) dim, buf);
            } else if (dim instanceof ComplexMatrixF) {
                printRowComplexF(row, _cols, (ComplexMatrixF) dim, buf);
            }
        }
        if (row == LAST_IDX && _rows < dim.numRows()) {
            int empty = _cols < dim.numColumns() ? MAX_ROWCOL : dim.numColumns();
            for (int i = 0; i < empty; ++i) {
                buf.append("......");
                if (i != empty - 1) {
                    buf.append(", ");
                }
            }
            buf.append(System.lineSeparator());
            if (dim instanceof MatrixD) {
                printRowD(dim.numRows() - 1, _cols, (MatrixD) dim, buf);
            } else if (dim instanceof MatrixF) {
                printRowF(dim.numRows() - 1, _cols, (MatrixF) dim, buf);
            } else if (dim instanceof ComplexMatrixD) {
                printRowComplexD(dim.numRows() - 1, _cols, (ComplexMatrixD) dim, buf);
            } else if (dim instanceof ComplexMatrixF) {
                printRowComplexF(dim.numRows() - 1, _cols, (ComplexMatrixF) dim, buf);
            }
        }
        return buf.toString();
    }

    private static void checkBigendian(boolean isBigendian) throws IOException {
        if (!isBigendian) {
            throw new IOException("Unexpected little endian storage format");
        }
    }

    private static void printRowD(int row, int _cols, MatrixD m, StringBuilder buf) {
        int col;
        for (col = 0; col < _cols; ++col) {
            buf.append(String.format(FORMAT_D, m.getUnsafe(row, col)));
            if (col < _cols - 1) {
                buf.append(", ");
            }
        }
        if (col == LAST_IDX && _cols < m.numColumns()) {
            buf.append(", ......, ");
            buf.append(String.format(FORMAT_D, m.getUnsafe(row, m.numColumns() - 1)));
        }
        buf.append(System.lineSeparator());
    }

    private static void printRowF(int row, int _cols, MatrixF m, StringBuilder buf) {
        int col;
        for (col = 0; col < _cols; ++col) {
            buf.append(String.format(FORMAT_F, m.getUnsafe(row, col)));
            if (col < _cols - 1) {
                buf.append(", ");
            }
        }
        if (col == LAST_IDX && _cols < m.numColumns()) {
            buf.append(", ......, ");
            buf.append(String.format(FORMAT_F, m.getUnsafe(row, m.numColumns() - 1)));
        }
        buf.append(System.lineSeparator());
    }

    private static void printRowComplexD(int row, int _cols, ComplexMatrixD m, StringBuilder buf) {
        int col;
        for (col = 0; col < _cols; ++col) {
            buf.append(m.getUnsafe(row, col).toString());
            if (col < _cols - 1) {
                buf.append(", ");
            }
        }
        if (col == LAST_IDX && _cols < m.numColumns()) {
            buf.append(", ......, ");
            buf.append(m.getUnsafe(row, m.numColumns() - 1).toString());
        }
        buf.append(System.lineSeparator());
    }

    private static void printRowComplexF(int row, int _cols, ComplexMatrixF m, StringBuilder buf) {
        int col;
        for (col = 0; col < _cols; ++col) {
            buf.append(m.getUnsafe(row, col).toString());
            if (col < _cols - 1) {
                buf.append(", ");
            }
        }
        if (col == LAST_IDX && _cols < m.numColumns()) {
            buf.append(", ......, ");
            buf.append(m.getUnsafe(row, m.numColumns() - 1).toString());
        }
        buf.append(System.lineSeparator());
    }

    static Lapack getLapack() {
        if (getBooleanPropVal(USE_NETLIB, false)) {
            return Lapack.getInstance(false);
        }
        return Lapack.getInstance();
    }

    static Blas getBlas() {
        if (getBooleanPropVal(USE_NETLIB, false)) {
            return Blas.getInstance(false);
        }
        return Blas.getInstance();
    }

    private static boolean getBooleanPropVal(String prop, boolean defVal) {
        boolean val = defVal;
        try {
            String s = System.getProperty(prop, Boolean.toString(defVal));
            val = Boolean.parseBoolean(s.trim());
        } catch (IllegalArgumentException | NullPointerException ignore) {
        }
        return val;
    }

    // 256k
    private static final int BUF_SIZE = 1 << 18;
    private static final int MAX_ROWCOL = 6;
    private static final int LAST_IDX = MAX_ROWCOL - 1;
    private static final String FORMAT_F = "%.8E";
    private static final String FORMAT_D = "%.12E";
    private static final String USE_NETLIB = "net.jamu.matrix.use.java.implementation";

    private Matrices() {
        throw new AssertionError();
    }
}
