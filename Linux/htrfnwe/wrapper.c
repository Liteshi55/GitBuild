#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <Python.h>
#include <numpy/arrayobject.h>
#include "halftrend.h"

static PyObject* py_halftrend(PyObject* self, PyObject* args) {
    PyObject *high_obj, *low_obj, *close_obj, *tr_obj;
    int amplitude;
    double channel_deviation;

    if (!PyArg_ParseTuple(args, "OOOOid", &high_obj, &low_obj, &close_obj, &tr_obj, &amplitude, &channel_deviation)) {
        return NULL;
    }

    PyArrayObject *high_array = (PyArrayObject*) PyArray_FROM_OTF(high_obj, NPY_FLOAT32, NPY_IN_ARRAY);
    PyArrayObject *low_array = (PyArrayObject*) PyArray_FROM_OTF(low_obj, NPY_FLOAT32, NPY_IN_ARRAY);
    PyArrayObject *close_array = (PyArrayObject*) PyArray_FROM_OTF(close_obj, NPY_FLOAT32, NPY_IN_ARRAY);
    PyArrayObject *tr_array = (PyArrayObject*) PyArray_FROM_OTF(tr_obj, NPY_FLOAT32, NPY_IN_ARRAY);

    if (high_array == NULL || low_array == NULL || close_array == NULL || tr_array == NULL) {
        Py_XDECREF(high_array);
        Py_XDECREF(low_array);
        Py_XDECREF(close_array);
        Py_XDECREF(tr_array);
        return NULL;
    }

    int n = (int)PyArray_DIM(high_array, 0);
    float *high = (float*) PyArray_DATA(high_array);
    float *low = (float*) PyArray_DATA(low_array);
    float *close = (float*) PyArray_DATA(close_array);
    float *tr = (float*) PyArray_DATA(tr_array);

    npy_intp dims[1] = {n};
    PyObject *halftrend_array = PyArray_SimpleNew(1, dims, NPY_FLOAT32);
    PyObject *atrHigh_array = PyArray_SimpleNew(1, dims, NPY_FLOAT32);
    PyObject *atrLow_array = PyArray_SimpleNew(1, dims, NPY_FLOAT32);
    PyObject *arrowUp_array = PyArray_SimpleNew(1, dims, NPY_FLOAT32);
    PyObject *arrowDown_array = PyArray_SimpleNew(1, dims, NPY_FLOAT32);
    PyObject *buy_array = PyArray_SimpleNew(1, dims, NPY_INT32);
    PyObject *sell_array = PyArray_SimpleNew(1, dims, NPY_INT32);

    if (halftrend_array == NULL || atrHigh_array == NULL || atrLow_array == NULL ||
        arrowUp_array == NULL || arrowDown_array == NULL || buy_array == NULL || sell_array == NULL) {
        Py_XDECREF(halftrend_array);
        Py_XDECREF(atrHigh_array);
        Py_XDECREF(atrLow_array);
        Py_XDECREF(arrowUp_array);
        Py_XDECREF(arrowDown_array);
        Py_XDECREF(buy_array);
        Py_XDECREF(sell_array);
        Py_XDECREF(high_array);
        Py_XDECREF(low_array);
        Py_XDECREF(close_array);
        Py_XDECREF(tr_array);
        return NULL;
    }

    halftrend(high, low, close, tr, n, amplitude, channel_deviation,
              (float*) PyArray_DATA((PyArrayObject*)halftrend_array),
              (float*) PyArray_DATA((PyArrayObject*)atrHigh_array),
              (float*) PyArray_DATA((PyArrayObject*)atrLow_array),
              (float*) PyArray_DATA((PyArrayObject*)arrowUp_array),
              (float*) PyArray_DATA((PyArrayObject*)arrowDown_array),
              (int*) PyArray_DATA((PyArrayObject*)buy_array),
              (int*) PyArray_DATA((PyArrayObject*)sell_array));

    PyObject *result = Py_BuildValue("{s:O, s:O, s:O, s:O, s:O, s:O, s:O}",
                                     "halftrend", halftrend_array,
                                     "atrHigh", atrHigh_array,
                                     "atrLow", atrLow_array,
                                     "arrowUp", arrowUp_array,
                                     "arrowDown", arrowDown_array,
                                     "buy", buy_array,
                                     "sell", sell_array);

    Py_DECREF(halftrend_array);
    Py_DECREF(atrHigh_array);
    Py_DECREF(atrLow_array);
    Py_DECREF(arrowUp_array);
    Py_DECREF(arrowDown_array);
    Py_DECREF(buy_array);
    Py_DECREF(sell_array);

    Py_DECREF(high_array);
    Py_DECREF(low_array);
    Py_DECREF(close_array);
    Py_DECREF(tr_array);

    return result;
}

static PyMethodDef HalfTrendMethods[] = {
    {"halftrend", py_halftrend, METH_VARARGS, "Calculate halftrend indicators"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef halftrendmodule = {
    PyModuleDef_HEAD_INIT,
    "halftrend",
    NULL,
    -1,
    HalfTrendMethods
};

PyMODINIT_FUNC PyInit_halftrend(void) {
    import_array();
    return PyModule_Create(&halftrendmodule);
}
