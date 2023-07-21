/*
 * Copyright 2020 Stefan Zobel
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
package net.jamu.complex;

/**
 * Utilities for arrays containing complex numbers.
 */
public final class ZArrayUtil {

    /**
     * Convert even length float[] array to corresponding {@link Zf} array.
     * 
     * @param a
     *            even length float[] array to convert
     * @return the resulting Zf[] array
     */
    public static Zf[] primitiveToComplexArray(float[] a) {
        if (a == null || a.length == 0) {
            return new Zf[] {};
        }
        checkEvenLength(a);
        Zf[] c = new Zf[a.length / 2];
        for (int i = 0; i < c.length; ++i) {
            int j = 2 * i;
            c[i] = new ZfImpl(a[j], a[j + 1]);
        }
        return c;
    }

    /**
     * Convert even length double[] array to corresponding {@link Zd} array.
     * 
     * @param a
     *            even length double[] array to convert
     * @return the resulting Zd[] array
     */
    public static Zd[] primitiveToComplexArray(double[] a) {
        if (a == null || a.length == 0) {
            return new Zd[] {};
        }
        checkEvenLength(a);
        Zd[] c = new Zd[a.length / 2];
        for (int i = 0; i < c.length; ++i) {
            int j = 2 * i;
            c[i] = new ZdImpl(a[j], a[j + 1]);
        }
        return c;
    }

    /**
     * Convert two float[] arrays separately containing the real and imaginary
     * part to corresponding {@link Zf} array.
     * 
     * @param re
     *            real parts
     * @param im
     *            imaginary parts
     * @return the resulting Zf[] array
     */
    public static Zf[] primitiveToComplexArray(float[] re, float[] im) {
        if (re == null || re.length == 0 || im == null || im.length == 0) {
            return new Zf[] {};
        }
        if (re.length != im.length) {
            throw new IllegalArgumentException(
                    "re[] and im[] arrays must have same length: " + re.length + " != " + im.length);
        }
        Zf[] c = new Zf[re.length];
        for (int i = 0; i < c.length; ++i) {
            c[i] = new ZfImpl(re[i], im[i]);
        }
        return c;
    }

    /**
     * Convert two double[] arrays separately containing the real and imaginary
     * part to corresponding {@link Zd} array.
     * 
     * @param re
     *            real parts
     * @param im
     *            imaginary parts
     * @return the resulting Zd[] array
     */
    public static Zd[] primitiveToComplexArray(double[] re, double[] im) {
        if (re == null || re.length == 0 || im == null || im.length == 0) {
            return new Zd[] {};
        }
        if (re.length != im.length) {
            throw new IllegalArgumentException(
                    "re[] and im[] arrays must have same length: " + re.length + " != " + im.length);
        }
        Zd[] c = new Zd[re.length];
        for (int i = 0; i < c.length; ++i) {
            c[i] = new ZdImpl(re[i], im[i]);
        }
        return c;
    }

    /**
     * Convert {@link Zf} array to an even length float[] array.
     * 
     * @param c
     *            complex array to convert
     * @return resulting even length float[] array
     */
    public static float[] complexToPrimitiveArray(Zf[] c) {
        if (c == null || c.length == 0) {
            return new float[] {};
        }
        float[] a = new float[2 * c.length];
        for (int i = 0; i < c.length; ++i) {
            Zf z = c[i];
            if (z != null) {
                int j = 2 * i;
                a[j] = z.re();
                a[j + 1] = z.im();
            }
        }
        return a;
    }

    /**
     * Convert {@link Zd} array to an even length double[] array.
     * 
     * @param c
     *            complex array to convert
     * @return resulting even length double[] array
     */
    public static double[] complexToPrimitiveArray(Zd[] c) {
        if (c == null || c.length == 0) {
            return new double[] {};
        }
        double[] a = new double[2 * c.length];
        for (int i = 0; i < c.length; ++i) {
            Zd z = c[i];
            if (z != null) {
                int j = 2 * i;
                a[j] = z.re();
                a[j + 1] = z.im();
            }
        }
        return a;
    }

    /**
     * Filter real parts from an even length float[] array.
     * 
     * @param a
     *            float[] array to filter
     * @return the real parts
     */
    public static float[] filterRealParts(float[] a) {
        if (a == null || a.length == 0) {
            return new float[] {};
        }
        checkEvenLength(a);
        float[] re = new float[a.length / 2];
        for (int i = 0; i < re.length; ++i) {
            re[i] = a[2 * i];
        }
        return re;
    }

    /**
     * Filter real parts from an even length double[] array.
     * 
     * @param a
     *            double[] array to filter
     * @return the real parts
     */
    public static double[] filterRealParts(double[] a) {
        if (a == null || a.length == 0) {
            return new double[] {};
        }
        checkEvenLength(a);
        double[] re = new double[a.length / 2];
        for (int i = 0; i < re.length; ++i) {
            re[i] = a[2 * i];
        }
        return re;
    }

    /**
     * Filter imaginary parts from an even length float[] array.
     * 
     * @param a
     *            float[] array to filter
     * @return the imaginary parts
     */
    public static float[] filterImaginaryParts(float[] a) {
        if (a == null || a.length == 0) {
            return new float[] {};
        }
        checkEvenLength(a);
        float[] im = new float[a.length / 2];
        for (int i = 0; i < im.length; ++i) {
            int j = 2 * i + 1;
            im[i] = a[j];
        }
        return im;
    }

    /**
     * Filter imaginary parts from an even length double[] array.
     * 
     * @param a
     *            double[] array to filter
     * @return the imaginary parts
     */
    public static double[] filterImaginaryParts(double[] a) {
        if (a == null || a.length == 0) {
            return new double[] {};
        }
        checkEvenLength(a);
        double[] im = new double[a.length / 2];
        for (int i = 0; i < im.length; ++i) {
            int j = 2 * i + 1;
            im[i] = a[j];
        }
        return im;
    }

    /**
     * Compute the L2 norm of an even length double[] vector (can also be used
     * for the computation of the Frobenius norm of complex matrices stored in
     * either column-major or row-major layout).
     * 
     * @param a
     *            an even length double[] vector
     * @return the L2 norm of the vector
     */
    public static double l2norm(double[] a) {// vectorization prospect
        if (a == null || a.length == 0) {
            return 0.0;
        }
        checkEvenLength(a);
        double scale = 0.0;
        for (int i = 0; i < a.length; i += 2) {
            double xr = a[i];
            double xi = a[i + 1]; // "lgtm[java/index-out-of-bounds]"
            if (xr != 0.0 || xi != 0.0) {
                scale = Math.max(scale, Math.abs(xr) + Math.abs(xi));
            }
        }
        if (scale == 0.0) {
            return 0.0;
        }
        while (scale <= 1.1) {
            scale = scale * 1000.0;
        }
        scale = 1.0 / scale;
        double sumsquared = 0.0;
        for (int i = 0; i < a.length; ++i) {
            double x = a[i];
            if (x != 0.0) {
                double scaled = scale * x;
                sumsquared += (scaled * scaled);
            }
        }
        return Math.sqrt(sumsquared) / scale;
    }

    /**
     * Compute the L2 norm of an even length float[] vector (can also be used
     * for the computation of the Frobenius norm of complex matrices stored in
     * either column-major or row-major layout).
     * 
     * @param a
     *            an even length float[] vector
     * @return the L2 norm of the vector
     */
    public static float l2norm(float[] a) {
        if (a == null || a.length == 0) {
            return 0.0f;
        }
        checkEvenLength(a);
        double scale = 0.0;
        for (int i = 0; i < a.length; i += 2) {
            double xr = a[i];
            double xi = a[i + 1]; // "lgtm[java/index-out-of-bounds]"
            if (xr != 0.0 || xi != 0.0) {
                scale = Math.max(scale, Math.abs(xr) + Math.abs(xi));
            }
        }
        if (scale == 0.0) {
            return 0.0f;
        }
        while (scale <= 1.1) {
            scale = scale * 1000.0;
        }
        scale = 1.0 / scale;
        double sumsquared = 0.0;
        for (int i = 0; i < a.length; ++i) {
            double x = a[i];
            if (x != 0.0) {
                double scaled = scale * x;
                sumsquared += (scaled * scaled);
            }
        }
        return (float) (Math.sqrt(sumsquared) / scale);
    }

    private static void checkEvenLength(float[] a) {
        if (a.length % 2 != 0) {
            throw new IllegalArgumentException("array length must be even: " + a.length);
        }
    }

    private static void checkEvenLength(double[] a) {
        if (a.length % 2 != 0) {
            throw new IllegalArgumentException("array length must be even: " + a.length);
        }
    }

    private ZArrayUtil() {
        //
    }
}
