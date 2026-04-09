export function foo(a: number, b: number)
: number {
    let x = a + b; let y = a - b;
    if ( x > y) {
        return a;
    } else {
        x = x + 1;
        if (x < 10) return b;
    }
}