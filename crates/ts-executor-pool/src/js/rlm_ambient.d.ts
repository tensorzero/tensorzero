// rlm_ambient.d.ts — Minimal ambient type declarations for the RLM sandbox.
//
// We use noLib:true for ~8x faster typechecking vs the full TypeScript stdlib
// (see benches/ts_checker.rs). This file provides the subset of ES built-ins
// that RLM-generated code actually needs.
//
// Available types and globals:
//
//   Compiler internals:  Function, CallableFunction, NewableFunction, IArguments
//   Primitives:          Boolean + BooleanConstructor, Number + NumberConstructor,
//                        String + StringConstructor (all usable as values)
//   Collections:         Array, ReadonlyArray, ArrayConstructor, Map, Set
//   Objects:             Object + ObjectConstructor, JSON, Math, Date, RegExp
//   Async:               Promise + PromiseConstructor, PromiseLike, Awaited
//   Iterators:           Iterator, Iterable, IterableIterator, Symbol
//   Errors:              Error, TypeError, RangeError, SyntaxError, ReferenceError
//   Global functions:    parseInt, parseFloat, isNaN, isFinite,
//                        encodeURIComponent, decodeURIComponent
//   Utility types:       Partial, Required, Readonly, Pick, Omit, Record,
//                        Exclude, Extract, NonNullable, ReturnType, Parameters
//   RLM sandbox:         context, FINAL, llm_query, llm_query_batched,
//                        console, globalThis

// Function types (required by TypeScript compiler internals)
interface Function {
  apply(this: Function, thisArg: any, argArray?: any): any;
  call(this: Function, thisArg: any, ...argArray: any[]): any;
  bind(this: Function, thisArg: any, ...argArray: any[]): any;
  toString(): string;
  prototype: any;
  readonly length: number;
}
interface FunctionConstructor {
  new (...args: string[]): Function;
  (...args: string[]): Function;
  readonly prototype: Function;
}
declare var Function: FunctionConstructor;

interface CallableFunction extends Function {}
interface NewableFunction extends Function {}

interface IArguments {
  [index: number]: any;
  length: number;
  callee: Function;
}

// Object (interface required alongside ObjectConstructor for global type)
interface Object {
  constructor: Function;
  toString(): string;
  valueOf(): Object;
  hasOwnProperty(v: PropertyKey): boolean;
}

// Primitive wrappers
interface Boolean {}
interface BooleanConstructor {
  new (value?: any): Boolean;
  <T>(value?: T): boolean;
}
declare var Boolean: BooleanConstructor;

interface Number {
  toString(radix?: number): string;
  toFixed(fractionDigits?: number): string;
}
interface NumberConstructor {
  new (value?: any): Number;
  (value?: any): number;
  readonly NaN: number;
  readonly POSITIVE_INFINITY: number;
  readonly NEGATIVE_INFINITY: number;
  readonly MAX_SAFE_INTEGER: number;
  readonly MIN_SAFE_INTEGER: number;
  isFinite(number: unknown): boolean;
  isInteger(number: unknown): boolean;
  isNaN(number: unknown): boolean;
  isSafeInteger(number: unknown): boolean;
}
declare var Number: NumberConstructor;

interface String {
  readonly length: number;
  charAt(pos: number): string;
  charCodeAt(index: number): number;
  indexOf(searchString: string, position?: number): number;
  lastIndexOf(searchString: string, position?: number): number;
  includes(searchString: string, position?: number): boolean;
  startsWith(searchString: string, position?: number): boolean;
  endsWith(searchString: string, endPosition?: number): boolean;
  slice(start?: number, end?: number): string;
  substring(start: number, end?: number): string;
  toLowerCase(): string;
  toUpperCase(): string;
  trim(): string;
  trimStart(): string;
  trimEnd(): string;
  split(separator: string | RegExp, limit?: number): string[];
  replace(searchValue: string | RegExp, replaceValue: string): string;
  replaceAll(searchValue: string | RegExp, replaceValue: string): string;
  match(regexp: string | RegExp): RegExpMatchArray | null;
  matchAll(regexp: RegExp): IterableIterator<RegExpMatchArray>;
  padStart(maxLength: number, fillString?: string): string;
  padEnd(maxLength: number, fillString?: string): string;
  repeat(count: number): string;
  concat(...strings: string[]): string;
  [index: number]: string;
}
interface StringConstructor {
  new (value?: any): String;
  (value?: any): string;
  fromCharCode(...codes: number[]): string;
}
declare var String: StringConstructor;

interface TemplateStringsArray extends ReadonlyArray<string> {
  readonly raw: readonly string[];
}

// Symbol
interface Symbol {
  readonly description: string | undefined;
  toString(): string;
  valueOf(): symbol;
}

interface SymbolConstructor {
  readonly iterator: unique symbol;
  readonly asyncIterator: unique symbol;
}
declare var Symbol: SymbolConstructor;

// Iterator protocols
interface IteratorYieldResult<TYield> {
  done?: false;
  value: TYield;
}
interface IteratorReturnResult<TReturn> {
  done: true;
  value: TReturn;
}
type IteratorResult<T, TReturn = any> =
  | IteratorYieldResult<T>
  | IteratorReturnResult<TReturn>;

interface Iterator<T, TReturn = any, TNext = any> {
  next(...args: [] | [TNext]): IteratorResult<T, TReturn>;
  return?(value?: TReturn): IteratorResult<T, TReturn>;
  throw?(e?: any): IteratorResult<T, TReturn>;
}

interface Iterable<T, TReturn = any, TNext = any> {
  [Symbol.iterator](): Iterator<T, TReturn, TNext>;
}

interface IterableIterator<T> extends Iterator<T> {
  [Symbol.iterator](): IterableIterator<T>;
}

// Array
interface ReadonlyArray<T> {
  readonly length: number;
  readonly [n: number]: T;
  [Symbol.iterator](): IterableIterator<T>;
  indexOf(searchElement: T, fromIndex?: number): number;
  lastIndexOf(searchElement: T, fromIndex?: number): number;
  includes(searchElement: T, fromIndex?: number): boolean;
  find(
    predicate: (value: T, index: number, obj: readonly T[]) => unknown
  ): T | undefined;
  findIndex(
    predicate: (value: T, index: number, obj: readonly T[]) => unknown
  ): number;
  every<S extends T>(
    predicate: (value: T, index: number, array: readonly T[]) => value is S
  ): this is readonly S[];
  every(
    predicate: (value: T, index: number, array: readonly T[]) => unknown
  ): boolean;
  some(
    predicate: (value: T, index: number, array: readonly T[]) => unknown
  ): boolean;
  forEach(
    callbackfn: (value: T, index: number, array: readonly T[]) => void
  ): void;
  map<U>(callbackfn: (value: T, index: number, array: readonly T[]) => U): U[];
  filter<S extends T>(
    predicate: (value: T, index: number, array: readonly T[]) => value is S
  ): S[];
  filter(
    predicate: (value: T, index: number, array: readonly T[]) => unknown
  ): T[];
  reduce(
    callbackfn: (
      previousValue: T,
      currentValue: T,
      currentIndex: number,
      array: readonly T[]
    ) => T
  ): T;
  reduce<U>(
    callbackfn: (
      previousValue: U,
      currentValue: T,
      currentIndex: number,
      array: readonly T[]
    ) => U,
    initialValue: U
  ): U;
  join(separator?: string): string;
  slice(start?: number, end?: number): T[];
  concat(...items: (T | ConcatArray<T>)[]): T[];
  flat<D extends number = 1>(depth?: D): FlatArray<T[], D>[];
  flatMap<U>(
    callback: (value: T, index: number, array: T[]) => U | ReadonlyArray<U>
  ): U[];
}

interface ConcatArray<T> {
  readonly length: number;
  readonly [n: number]: T;
  join(separator?: string): string;
  slice(start?: number, end?: number): T[];
}

type FlatArray<Arr, Depth extends number> = {
  done: Arr;
  recur: Arr extends ReadonlyArray<infer InnerArr>
    ? FlatArray<
        InnerArr,
        [-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20][Depth]
      >
    : Arr;
}[Depth extends -1 ? "done" : "recur"];

interface Array<T> extends ReadonlyArray<T> {
  length: number;
  [n: number]: T;
  push(...items: T[]): number;
  pop(): T | undefined;
  shift(): T | undefined;
  unshift(...items: T[]): number;
  splice(start: number, deleteCount?: number, ...items: T[]): T[];
  sort(compareFn?: (a: T, b: T) => number): this;
  reverse(): T[];
  fill(value: T, start?: number, end?: number): this;
}

interface ArrayConstructor {
  new <T>(...items: T[]): T[];
  isArray(arg: any): arg is any[];
  from<T>(arrayLike: ArrayLike<T>): T[];
  from<T, U>(
    arrayLike: ArrayLike<T>,
    mapfn: (v: T, k: number) => U
  ): U[];
}
declare var Array: ArrayConstructor;

interface ArrayLike<T> {
  readonly length: number;
  readonly [n: number]: T;
}

// Object
interface ObjectConstructor {
  keys(o: object): string[];
  values(o: object): any[];
  entries(o: object): [string, any][];
  assign<T extends object>(target: T, ...sources: any[]): T;
  freeze<T>(obj: T): Readonly<T>;
  fromEntries<T = any>(
    entries: Iterable<readonly [PropertyKey, T]>
  ): { [k: string]: T };
  defineProperty(
    o: any,
    p: PropertyKey,
    attributes: PropertyDescriptor
  ): any;
  getOwnPropertyNames(o: any): string[];
  hasOwn(o: object, v: PropertyKey): boolean;
  create(o: object | null, properties?: PropertyDescriptorMap): any;
}
declare var Object: ObjectConstructor;

interface PropertyDescriptor {
  configurable?: boolean;
  enumerable?: boolean;
  value?: any;
  writable?: boolean;
  get?(): any;
  set?(v: any): void;
}

interface PropertyDescriptorMap {
  [key: string]: PropertyDescriptor;
}

type PropertyKey = string | number | symbol;

// JSON
interface JSON {
  parse(text: string, reviver?: (key: string, value: any) => any): any;
  stringify(
    value: any,
    replacer?: ((key: string, value: any) => any) | (number | string)[] | null,
    space?: string | number
  ): string;
}
declare var JSON: JSON;

// Math
interface Math {
  readonly PI: number;
  abs(x: number): number;
  ceil(x: number): number;
  floor(x: number): number;
  round(x: number): number;
  max(...values: number[]): number;
  min(...values: number[]): number;
  pow(x: number, y: number): number;
  sqrt(x: number): number;
  random(): number;
  log(x: number): number;
  log2(x: number): number;
  log10(x: number): number;
  trunc(x: number): number;
  sign(x: number): number;
}
declare var Math: Math;

// Promise
interface PromiseLike<T> {
  then<TResult1 = T, TResult2 = never>(
    onfulfilled?:
      | ((value: T) => TResult1 | PromiseLike<TResult1>)
      | null,
    onrejected?:
      | ((reason: any) => TResult2 | PromiseLike<TResult2>)
      | null
  ): PromiseLike<TResult1 | TResult2>;
}

interface Promise<T> {
  then<TResult1 = T, TResult2 = never>(
    onfulfilled?:
      | ((value: T) => TResult1 | PromiseLike<TResult1>)
      | null,
    onrejected?:
      | ((reason: any) => TResult2 | PromiseLike<TResult2>)
      | null
  ): Promise<TResult1 | TResult2>;
  catch<TResult = never>(
    onrejected?:
      | ((reason: any) => TResult | PromiseLike<TResult>)
      | null
  ): Promise<T | TResult>;
  finally(onfinally?: (() => void) | null): Promise<T>;
}

interface PromiseConstructor {
  new <T>(
    executor: (
      resolve: (value: T | PromiseLike<T>) => void,
      reject: (reason?: any) => void
    ) => void
  ): Promise<T>;
  resolve<T>(value: T | PromiseLike<T>): Promise<T>;
  reject<T = never>(reason?: any): Promise<T>;
  all<T extends readonly unknown[]>(
    values: T
  ): Promise<{ -readonly [P in keyof T]: Awaited<T[P]> }>;
  allSettled<T extends readonly unknown[]>(
    values: T
  ): Promise<{
    -readonly [P in keyof T]: PromiseSettledResult<Awaited<T[P]>>;
  }>;
  race<T extends readonly unknown[]>(
    values: T
  ): Promise<Awaited<T[number]>>;
}
declare var Promise: PromiseConstructor;

type Awaited<T> = T extends null | undefined
  ? T
  : T extends object & { then(onfulfilled: infer F, ...args: infer _): any }
    ? F extends (value: infer V, ...args: infer _) => any
      ? Awaited<V>
      : never
    : T;

interface PromiseFulfilledResult<T> {
  status: "fulfilled";
  value: T;
}
interface PromiseRejectedResult {
  status: "rejected";
  reason: any;
}
type PromiseSettledResult<T> =
  | PromiseFulfilledResult<T>
  | PromiseRejectedResult;

// RegExp
interface RegExp {
  exec(string: string): RegExpExecArray | null;
  test(string: string): boolean;
  readonly source: string;
  readonly flags: string;
  readonly global: boolean;
  readonly ignoreCase: boolean;
  readonly multiline: boolean;
  lastIndex: number;
}
interface RegExpMatchArray extends Array<string> {
  index?: number;
  input?: string;
  groups?: { [key: string]: string };
}
interface RegExpExecArray extends Array<string> {
  index: number;
  input: string;
  groups?: { [key: string]: string };
}
interface RegExpConstructor {
  new (pattern: string | RegExp, flags?: string): RegExp;
  (pattern: string | RegExp, flags?: string): RegExp;
}
declare var RegExp: RegExpConstructor;

// Error
interface Error {
  name: string;
  message: string;
  stack?: string;
}
interface ErrorConstructor {
  new (message?: string): Error;
  (message?: string): Error;
}
declare var Error: ErrorConstructor;
declare var TypeError: ErrorConstructor;
declare var RangeError: ErrorConstructor;
declare var SyntaxError: ErrorConstructor;
declare var ReferenceError: ErrorConstructor;

// Map and Set
interface Map<K, V> {
  readonly size: number;
  get(key: K): V | undefined;
  set(key: K, value: V): this;
  has(key: K): boolean;
  delete(key: K): boolean;
  clear(): void;
  forEach(
    callbackfn: (value: V, key: K, map: Map<K, V>) => void
  ): void;
  keys(): IterableIterator<K>;
  values(): IterableIterator<V>;
  entries(): IterableIterator<[K, V]>;
  [Symbol.iterator](): IterableIterator<[K, V]>;
}
interface MapConstructor {
  new <K, V>(entries?: readonly (readonly [K, V])[] | null): Map<K, V>;
}
declare var Map: MapConstructor;

interface Set<T> {
  readonly size: number;
  add(value: T): this;
  has(value: T): boolean;
  delete(value: T): boolean;
  clear(): void;
  forEach(
    callbackfn: (value: T, value2: T, set: Set<T>) => void
  ): void;
  keys(): IterableIterator<T>;
  values(): IterableIterator<T>;
  entries(): IterableIterator<[T, T]>;
  [Symbol.iterator](): IterableIterator<T>;
}
interface SetConstructor {
  new <T>(values?: readonly T[] | null): Set<T>;
}
declare var Set: SetConstructor;

// Date
interface Date {
  getTime(): number;
  toISOString(): string;
  toJSON(): string;
  toString(): string;
}
interface DateConstructor {
  new (): Date;
  new (value: number | string): Date;
  now(): number;
  parse(s: string): number;
}
declare var Date: DateConstructor;

// Global functions
declare function parseInt(string: string, radix?: number): number;
declare function parseFloat(string: string): number;
declare function isNaN(number: number): boolean;
declare function isFinite(number: number): boolean;
declare function encodeURIComponent(
  uriComponent: string | number | boolean
): string;
declare function decodeURIComponent(encodedURIComponent: string): string;
// Utility types used by TypeScript
type Partial<T> = { [P in keyof T]?: T[P] };
type Required<T> = { [P in keyof T]-?: T[P] };
type Readonly<T> = { readonly [P in keyof T]: T[P] };
type Pick<T, K extends keyof T> = { [P in K]: T[P] };
type Omit<T, K extends keyof any> = Pick<T, Exclude<keyof T, K>>;
type Record<K extends keyof any, T> = { [P in K]: T };
type Exclude<T, U> = T extends U ? never : T;
type Extract<T, U> = T extends U ? T : never;
type NonNullable<T> = T & {};
type ReturnType<T extends (...args: any) => any> = T extends (
  ...args: any
) => infer R
  ? R
  : any;
type Parameters<T extends (...args: any) => any> = T extends (
  ...args: infer P
) => any
  ? P
  : never;

// === RLM Sandbox Globals ===

/** The full data to process, available as a string. Almost always JSON. */
declare var context: string;

/**
 * Mark a value as the final answer and end processing.
 * Pass a string, object, or array (objects/arrays are JSON-serialized).
 */
declare function FINAL(
  value: string | Record<string, unknown> | unknown[]
): void;

/**
 * Send a prompt to an analysis LLM and get back a text response.
 * At shallow recursion depth, this spawns a child RLM loop.
 * At maximum depth, it makes a direct single-shot LLM call.
 */
declare function llm_query(prompt: string): Promise<string>;

/**
 * Send multiple prompts concurrently. More efficient than sequential
 * llm_query calls when you have independent sub-tasks.
 */
declare function llm_query_batched(prompts: string[]): Promise<string[]>;

/** Console for logging intermediate results. */
declare var console: {
  log(...args: unknown[]): void;
};
