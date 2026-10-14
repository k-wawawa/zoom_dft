# zoom_dft

## なにこれ

- CZTを用いて任意の等間隔グリッドで離散フーリエ・サイン・コサイン変換を行う
- 同じ変換条件を繰り返す場合は`ZoomDftPlan`の再利用が可能
- 一般的な DST-I / DST-II そのものではない

## インストール方法

``` toml
[dependencies]
zoom_dft = { git = "https://github.com/k-wawawa/zoom_dft.git" }
```

## 使い方

最小の使用例
``` Rust
use ndarray::array;
use zoom_dft::{GridEndpoint, ZoomDftSpec};

fn main() -> Result<(), Box<dyn std::error::Error>>{
    let input = array![1.0, 0.5, -0.25, 2.0];
    let spec = ZoomDftSpec::new()
        .i_range(0.0, 10.0)
        .o_range(0.0, 20.0)
        .i_grid(GridEndpoint::Included, 0.0)
        .o_grid(GridEndpoint::Included, 1.0);
    let plan = spec.plan(input.len(), 1000)?;
    let output = plan.zdst(input.view())?;
    Ok(())
}
```

プランの再利用例
``` Rust
let plan = spec.plan(input1.len(), output_len)?;

let output1 = plan.zdst(input1.view())?;
let output2 = plan.zdct(input2.view())?;
let output3 = plan.zdft(input3.view())?;
```

入力点数・出力点数・入出力グリッド・ZoomDstSpecの条件が同じ場合だけ再利用可能。

## グリッドの定義

### 入力座標
```
x[j] = i_min + Δi*(j + i_index_shift)
```

### 出力座標
```
y[k] = o_min + Δo * (k + o_index_shift)
```

### GridEndpointの意味

- Included: maxを刻み幅の計算に含める
- Excluded: maxは範囲の終端であり、座標点には含めない

### shiftの意味

index_shift = 1.0なら、最初の座標が1刻み分ずれる。  
index_shift = 0.5 かつ GridEndpoint::Excludedの場合、DSTの中間点でやるやつみたいな感じのグリッドになる

## 数学的な定義

この変換はrを入力側変数、qを出力側変数として

```
output[k] = Δr * Σ input[j] f(r[j] q[k])
f = sin(qr) or cos(qr) or exp(iqr)
```
を計算している。
特に

- πは自動でかからない
- 入力刻み幅は結果に含まれる
- 出力はzdftは複素数、zdst,zdctは実数
- 入出力グリッドは等間隔

である

## 出力座標の取得

``` Rust
let output_x = spec.xs(output_len)?;
```

## 1回だけ変換する方法

``` Rust
let output = spec.zdst(input.view(), output_len)?;
```

## API一覧

| API | 用途 |
|---|---|
| `ZoomDftSpec` | 入出力グリッドの定義(入力・出力範囲は実空間単位) |
| `ZoomDftSpec::zdft, zdst, zdct` | 一回だけの変換 |
| `ZoomDftSpec::plan` | 再利用可能な変換Planの作成 |
| `ZoomDftPlan::zdft, zdst, zdct` | 入力データの変換 |
| `GridEndpoint` | 最大端点を含むかどうかの指定 |

## 制約とエラー

- 入力点数は2以上
- 出力点数は2以上
- `ZoomDftPlan::zdft, zdst, zdct`の入力長はPlan作成時と一致する必要がある
- 入力・出力は等間隔

## テスト方法

```
cargo test
```
