extends CharacterBody2D

const SPEED = 300.0

var direction: Vector2


func _ready():
	direction = Vector2(randf_range(-1,1),randf_range(-1,1)).normalized()

func _physics_process(delta):
	# Get the input direction and handle the movement/deceleration.
	# As good practice, you should replace UI actions with custom gameplay actions.
	velocity = direction * SPEED
	move_and_slide()
	
	$Sprite2D.look_at(position + direction)

func hit_side_wall(_param: Area2D):
	direction.x = -direction.x

func hit_top_bottom_wall(_param: Area2D):
	direction.y = -direction.y
