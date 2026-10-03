/* SPDX-License-Identifier: MIT */

const assert = require('node:assert/strict');
const {spawnSync} = require('node:child_process');
const braces = require('braces');
const compile = require('braces/lib/compile');
const expand = require('braces/lib/expand');
const parse = require('braces/lib/parse');
const stringify = require('braces/lib/stringify');
const serialize = require('serialize-javascript');
const {v4: uuidv4, v5: uuidv5} = require('uuid');

assert.equal(require('braces/package.json').version, '3.0.4-fmmax.0');

const nestedAst = (depth) => {
  let ast = {type: 'text', value: 'a'};
  Array.from({length: depth}).forEach(() => {
    ast = {type: 'brace', nodes: [ast]};
  });
  return {type: 'root', nodes: [ast]};
};

const nestedPattern = (depth) => `${'{'.repeat(depth)}a,b${'}'.repeat(depth)}`;

// Preserve representative published behavior and the configured boundary.
assert.deepEqual(braces.expand('a/{b,c}/d'), ['a/b/d', 'a/c/d']);
assert.doesNotThrow(() => braces(nestedPattern(100)));
assert.throws(() => braces(nestedPattern(101)), /exceeds max depth \(100\)/);
assert.throws(() => braces(nestedPattern(1000)), /exceeds max depth \(100\)/);
assert.throws(
  () => parse('('.repeat(101) + ')'.repeat(101)),
  /exceeds max depth/,
);
assert.throws(
  () => braces('{{a,b},c}', {maxDepth: 1.9}),
  /exceeds max depth \(1\)/,
);
assert.throws(() => braces('{a,b}', {maxDepth: -1}), /exceeds max depth \(0\)/);

// Guard direct AST callers as well as string input.
[compile, expand, stringify].forEach((walk) => {
  assert.throws(() => walk(nestedAst(101)), /exceeds max depth \(100\)/);
  assert.throws(() => walk(nestedAst(10000)), /exceeds max depth \(100\)/);
});

// The depth-only backport must not change escapeInvalid output.
['{{a}}', '{a,{b}}', '{{x}y}', '{a,{b,{c}}', '{}{a}', '{1..8}'].forEach(
  (pattern) => {
    assert.equal(stringify(parse(pattern), {escapeInvalid: true}), pattern);
  },
);

// Exercise the injection validation added by the serialize-javascript fixes.
const fakeRegex = Object.create(RegExp.prototype);
Object.defineProperties(fakeRegex, {
  source: {get: () => 'x'},
  flags: {get: () => 'g");0;//'},
});
fakeRegex.toJSON = () => '@placeholder';
assert.equal(serialize({regex: fakeRegex}), '{"regex":new RegExp("x", "g")}');

const fakeDate = Object.create(Date.prototype);
fakeDate.toISOString = () => '");throw new Error("injected");//';
fakeDate.toJSON = () => '2024-01-01';
assert.throws(() => serialize({date: fakeDate}), /Invalid Date ISO string/);

// Run the array-like CPU-exhaustion reproducer in a bounded child process.
const arrayLikeProbe = spawnSync(
  process.execPath,
  [
    '-e',
    "const serialize=require('serialize-javascript');" +
      'const value=Object.create(Array.prototype);' +
      'value.length=2**32-1;' +
      'serialize({value});',
  ],
  {timeout: 5000},
);
assert.equal(
  arrayLikeProbe.status,
  0,
  arrayLikeProbe.error?.message || arrayLikeProbe.stderr.toString(),
);

// Confirm that the remaining overrides expose the APIs used downstream.
assert.match(serialize({expression: /a/gi}), /new RegExp/);
assert.match(uuidv4(), /^[0-9a-f-]{36}$/);
assert.throws(
  () => uuidv5('x', uuidv5.DNS, new Uint8Array(8), 4),
  /out of buffer bounds/,
);

import('update-notifier').then((module) => {
  assert.equal(typeof module.default, 'function');
});
