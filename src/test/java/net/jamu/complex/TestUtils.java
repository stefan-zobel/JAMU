/*
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
/*
 * Any changes, bugfixes or additions made by the maintainers
 * of the https://github.com/stefan-zobel/FFT library are
 * licensed under the Apache License, Version 2.0, as explained
 * at http://www.apache.org/licenses/LICENSE-2.0
 */
package net.jamu.complex;

import java.util.Random;

import org.junit.Assert;

public final class TestUtils {

    private static final Random rng = new Random();

    /**
     * Verifies that expected and actual are within delta, or are both NaN or
     * infinities of the same sign.
     */
    public static void assertEquals(double expected, double actual, double delta) {
        Assert.assertEquals(null, expected, actual, delta);
    }

    /**
     * Verifies that expected and actual are within delta, or are both NaN or
     * infinities of the same sign.
     */
    public static void assertEquals(String msg, double expected, double actual, double delta) {
        // check for NaN
        if (Double.isNaN(expected)) {
            Assert.assertTrue("" + actual + " is not NaN.", Double.isNaN(actual));
        } else {
            Assert.assertEquals(msg, expected, actual, delta);
        }
    }

    /**
     * Verifies that the two arguments are exactly the same, either both NaN or
     * infinities of same sign, or identical floating point values.
     */
    public static void assertSame(double expected, double actual) {
        Assert.assertEquals(expected, actual, 0);
    }

    /**
     * Verifies that real and imaginary parts of the two complex arguments are
     * exactly the same. Also ensures that NaN / infinite components match.
     */
    public static void assertSame(Zd expected, Zd actual) {
        assertSame(expected.re(), actual.re());
        assertSame(expected.im(), actual.im());
    }

    /**
     * Verifies that real and imaginary parts of the two complex arguments
     * differ by at most delta. Also ensures that NaN / infinite components
     * match.
     */
    public static void assertEquals(Zd expected, Zd actual, double delta) {
        Assert.assertEquals(expected.re(), actual.re(), delta);
        Assert.assertEquals(expected.im(), actual.im(), delta);
    }

    /**
     * Verifies that the relative error in actual versus expected is less than
     * or equal to relativeError. If expected is infinite or NaN, actual must be
     * the same (NaN or infinity of the same sign).
     *
     * @param expected
     *            expected value
     * @param actual
     *            observed value
     * @param relativeError
     *            maximum allowable relative error
     */
    public static void assertRelativelyEquals(double expected, double actual, double relativeError) {
        assertRelativelyEquals(null, expected, actual, relativeError);
    }

    /**
     * Verifies that the relative error in actual versus expected is less than
     * or equal to relativeError. If expected is infinite or NaN, actual must be
     * the same (NaN or infinity of the same sign).
     *
     * @param msg
     *            message to return with failure
     * @param expected
     *            expected value
     * @param actual
     *            observed value
     * @param relativeError
     *            maximum allowable relative error
     */
    public static void assertRelativelyEquals(String msg, double expected, double actual, double relativeError) {
        if (Double.isNaN(expected)) {
            Assert.assertTrue(msg, Double.isNaN(actual));
        } else if (Double.isNaN(actual)) {
            Assert.assertTrue(msg, Double.isNaN(expected));
        } else if (Double.isInfinite(actual) || Double.isInfinite(expected)) {
            Assert.assertEquals(expected, actual, relativeError);
        } else if (expected == 0.0) {
            Assert.assertEquals(msg, actual, expected, relativeError);
        } else {
            double absError = Math.abs(expected) * relativeError;
            Assert.assertEquals(msg, expected, actual, absError);
        }
    }

    // each individual double lies in [-1, 1)
    public static double[] randomData(int length) {
        double[] rand = new double[length];
        for (int i = 0; i < rand.length; ++i) {
            rand[i] = (rng.nextDouble() * 2.0) - 1.0;
        }
        return rand;
    }

    // range is [3, 8193]
    public static int randLengthOdd() {
        int len = -1;
        do {
            len = rng.nextInt(8194);
        } while (len <= 2 || (len % 2 == 0));
        return len;
    }

    // range is [6, 8194]
    public static int randLengthEvenNotPowerOf2() {
        int len = -1;
        do {
            len = rng.nextInt(8195);
        } while (len <= 2 || (len % 2 != 0) || (isPowerOfTwo(len)));
        return len;
    }

    public static boolean isPowerOfTwo(int n) {
        return (n > 0) && ((n & (n - 1)) == 0);
    }

    private TestUtils() {
        throw new AssertionError();
    }
}
