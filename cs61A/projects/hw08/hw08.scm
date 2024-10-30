(define (my-filter pred s)
  (cond 
    ((null? s)
     s)
    ((pred (car s))
     (cons (car s) (my-filter pred (cdr s))))
    (else
     (my-filter pred (cdr s)))))

(define (interleave lst1 lst2)
  (cond 
    ((null? lst1)
     lst2)
    ((null? lst2)
     lst1)
    (else
     (cons (car lst1) (interleave lst2 (cdr lst1))))))

(define (accumulate joiner start n term)
  (cond 
    ((not (= start 0))
     (joiner start (accumulate joiner 0 n term)))
    ((= n 1)
     (term n))
    (else
     (joiner (term n)
             (accumulate joiner 0 (- n 1) term)))))

(define (equ s num)
  (cond 
    ((null? s)       #t)
    ((= (car s) num) #f)
    (else            (equ (cdr s) num))))

(define (no-repeats lst)
  (cond 
    ((null? lst)
     nil)
    (else
     (cons (car lst)
           (my-filter (lambda (x) (not (= x (car lst))))
                      (no-repeats (cdr lst)))))))
