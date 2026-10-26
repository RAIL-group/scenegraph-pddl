(define (domain procthor)
  (:requirements :strips :equality)
  (:predicates 
               (Robot ?r)
               (Hand ?h)
               (Object ?b)
               (HandEmpty ?r ?h)
               (Holding ?r ?b ?h)
               (In ?b ?s)
               (At ?r ?s)
               (Motion ?s1 ?s2)
               (ContainerUnsafe ?s)
  )

  (:functions
    (Cost)
    (MotionCost ?s1 ?s2)
  )

  (:action pick
    :parameters (?r ?b ?s ?h)
    :precondition (and (Robot ?r) (Hand ?h) (HandEmpty ?r ?h) (At ?r ?s) (In ?b ?s))
    :effect (and (Holding ?r ?b ?h) (not (In ?b ?s))
                 (not (HandEmpty ?r ?h))
                 (increase (total-cost) 10) 
                 ))

  (:action place
    :parameters (?r ?b ?s ?h)
    :precondition (and (Robot ?r) (Hand ?h) (At ?r ?s) (Holding ?r ?b ?h) (not (ContainerUnsafe ?s)))
    :effect (and (HandEmpty ?r ?h) (In ?b ?s)
                 (not (Holding ?r ?b ?h))
                 (increase (total-cost) 10)
                 ))

  (:action move
    :parameters (?r ?s1 ?s2)
    :precondition (and (Robot ?r) (At ?r ?s1) (Motion ?s1 ?s2))
    :effect (and (At ?r ?s2) (not (At ?r ?s1))
                 (increase (total-cost) (MotionCost ?s1 ?s2))
            ))
)