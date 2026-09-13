#!/bin/sh
# Собирает index.html из трёх частей. Формулы отдельно, чтобы их можно было тестировать в node.
cd "$(dirname "$0")"
{ cat ui-head.html; echo '<script id="calc">'; cat calc-block.js; echo '</script>'; cat ui-tail.html; } > index.html
node test.js
