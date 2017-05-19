#ifndef DYNAMIC_KD_TREE
#define DYNAMIC_KD_TREE
#include <algorithm>
#include <cmath>
#include <queue>
#include <vector>
#define kd 2
class kd_tree {
public:
  struct point {
    double d[kd];
    point() {}
    point(double x, double y) {
      d[0] = x, d[1] = y;
    }
    inline double dist(const point &x) const {
      // specific to 2d points
      return hypot(d[0] - x.d[0], d[1] - x.d[1]);
    }
    inline bool operator<(const point &b) const { return d[0] < b.d[0]; }
  };

private:
  struct node {
    node *l, *r;
    point pid;
    int s;
    node(const point &p) : l(0), r(0), pid(p), s(1) {}
    inline void up() { s = (l ? l->s : 0) + 1 + (r ? r->s : 0); }
  } *root;
  const double alpha, loga;
  const double INF; //記得要給INF，表示極大值
  std::vector<node *> A;
  struct cmp {
    int sort_id;
    cmp(int id) : sort_id(id) {}
    inline bool operator()(const node *x, const node *y) const {
      return x->pid.d[sort_id] < y->pid.d[sort_id];
    }
  };
  void clear(node *o) {
    if (!o)
      return;
    clear(o->l);
    clear(o->r);
    delete o;
  }
  inline int size(node *o) { return o ? o->s : 0; }
  node *build(int k, int l, int r) {
    if (l > r)
      return 0;
    if (k == kd)
      k = 0;
    int mid = (l + r) / 2;
    std::nth_element(A.begin() + l, A.begin() + mid, A.begin() + r + 1, cmp(k));
    node *ret = A[mid];
    ret->l = build(k + 1, l, mid - 1);
    ret->r = build(k + 1, mid + 1, r);
    ret->up();
    return ret;
  }
  inline bool isbad(node *o) {
    return size(o->l) > alpha * o->s || size(o->r) > alpha * o->s;
  }
  void flatten(node *u, typename std::vector<node *>::iterator &it) {
    if (!u)
      return;
    flatten(u->l, it);
    *it = u;
    flatten(u->r, ++it);
  }
  bool insert(node *&u, int k, const point &x, int dep) {
    if (!u) {
      u = new node(x);
      return dep <= 0;
    }
    ++u->s;
    if (insert(x.d[k] < u->pid.d[k] ? u->l : u->r, (k + 1) % kd, x, dep - 1)) {
      if (!isbad(u))
        return 1;
      if ((int)A.size() < u->s)
        A.resize(u->s);
      auto it = A.begin();
      flatten(u, it);
      u = build(k, 0, u->s - 1);
    }
    return 0;
  }
  inline double heuristic(const double h[]) const {
    // specific to 2d point
    return hypot(h[0], h[1]);
  }
  void nearest(node *u, int k, const point &x, double *h, double &mndist) {
    if (u == 0 || heuristic(h) >= mndist)
      return;
    double dist = u->pid.dist(x);
    int old = h[k];
    mndist=std::min(mndist,dist);
    if (x.d[k] < u->pid.d[k]) {
      nearest(u->l, (k + 1) % kd, x, h, mndist);
      h[k] = x.d[k] - u->pid.d[k];
      nearest(u->r, (k + 1) % kd, x, h, mndist);
    } else {
      nearest(u->r, (k + 1) % kd, x, h, mndist);
      h[k] = x.d[k] - u->pid.d[k];
      nearest(u->l, (k + 1) % kd, x, h, mndist);
    }
    h[k] = old;
  }

public:
  kd_tree(const double &INF = 1e18, double a = 0.75)
      : root(0), alpha(a), loga(-log2(a)), INF(INF) {}
  inline void clear() { clear(root), root = 0; }
  inline void build(int n, const point *p) {
    clear(root), A.resize(n);
    for (int i = 0; i < n; ++i)
      A[i] = new node(p[i]);
    root = build(0, 0, n - 1);
  }
  inline void insert(const point &x) {
    insert(root, 0, x, log2(size(root)) / loga);
  }
  inline double nearest(const point &x) {
    // returns the distance to the nearest point
    double mndist = INF, h[kd] = {};
    nearest(root, 0, x, h, mndist);
    return mndist;
  }
  inline int size() { return root ? root->s : 0; }
};
#endif

