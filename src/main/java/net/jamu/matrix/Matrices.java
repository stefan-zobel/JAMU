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
import net.jamu.complex.Zf;

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
        for (int i = 0; i < 2 * data.length; ++i) {
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
        for (int i = 0; i < 2 * data.length; ++i) {
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
